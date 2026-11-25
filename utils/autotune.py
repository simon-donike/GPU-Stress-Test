"""Batch-size calibration using isolated executions of the real runner.

The parent deliberately never imports PyTorch or initializes CUDA. Candidate
processes own CUDA state, allocator caps, and the unchanged model workload.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from utils.config import StressTestConfig, default_run_name
from utils.tuning_analysis import analyze_trial, select_recommendation

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class AutotuneConfig:
    max_batch_size: int = 4096
    max_trials: int = 8
    seconds: float = 8.0
    timeout_seconds: float = 120.0
    target_utilization: float = 95.0
    memory_fraction: float = 0.85

    def validate(self, config: StressTestConfig) -> None:
        if config.accelerator != "gpu":
            raise ValueError("Autotuning requires GPU mode.")
        if isinstance(self.max_batch_size, bool) or not isinstance(self.max_batch_size, int) or self.max_batch_size < 1:
            raise ValueError("tune-max-batch-size must be a positive integer.")
        if config.batch_size > self.max_batch_size:
            raise ValueError("batch-size must not exceed tune-max-batch-size.")
        if isinstance(self.max_trials, bool) or not isinstance(self.max_trials, int) or not 1 <= self.max_trials <= 64:
            raise ValueError("tune-max-trials must be between 1 and 64.")
        for name in ("seconds", "timeout_seconds", "target_utilization", "memory_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Autotune {name} must be finite.")
        if not 3 <= self.seconds <= 60:
            raise ValueError("tune-seconds must be between 3 and 60 measured seconds per candidate.")
        if self.timeout_seconds < self.seconds + 10 or self.timeout_seconds > 600:
            raise ValueError("tune-timeout-seconds must allow 10 seconds beyond measurement and be <= 600.")
        if not 0 < self.target_utilization <= 100:
            raise ValueError("tune-target-utilization must be in (0, 100].")
        if not 0 < self.memory_fraction <= .95:
            raise ValueError("tune-memory-fraction must be in (0, 0.95].")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def command_for_config(config: StressTestConfig, python: str = sys.executable) -> list[str]:
    """Build argv without a shell; preserve the exact model, precision, and device selection."""
    # Every generated invocation already has its batch size selected. This also
    # prevents recursive calibration in discovery/trials and Lightning workers.
    args = [python, str(PROJECT_ROOT / "run.py"), "--no-autotune"]
    for key, value in config.to_dict().items():
        if value is None:
            continue
        if key == "generate_data_on_device":
            args.append("--generate-on-device" if value else "--no-generate-on-device")
        elif key == "generate_report":
            if not value:
                args.append("--no-report")
        else:
            name = "metrics-dir" if key == "metrics_root" else key.replace("_", "-")
            args.extend((f"--{name}", str(value)))
    return args


def _stop_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before its DDP children. Always clean the group too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _run_child(command: list[str], log: Path, timeout: float) -> tuple[int, bool]:
    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"Calibration parent received signal {signum}.")

    # Children have a separate session, so cancellation of the parent must
    # explicitly clean them up on SIGTERM as well as Ctrl+C.
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            timed_out = False
            try:
                try:
                    return_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _stop_group(process)
                    return_code = process.returncode or -signal.SIGKILL
            finally:
                _stop_group(process)
    finally:
        signal.signal(signal.SIGTERM, previous)
    return return_code, timed_out


def _uuid(value: Any) -> str:
    return str(value).strip().lower().removeprefix("gpu-")


def _smi(query: str, category: str = "gpu") -> list[list[str]]:
    result = subprocess.run(["nvidia-smi", f"--query-{category}={query}", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=5)
    return [[value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row]


def _check_idle(selected: set[str]) -> None:
    visible = {_uuid(row[0]) for row in _smi("uuid")}
    if not selected.issubset(visible):
        raise RuntimeError("Selected CUDA UUIDs lack physical-GPU telemetry. MIG autotuning is not supported.")
    busy = [row for row in _smi("gpu_uuid,pid", "compute-apps") if len(row) >= 2 and _uuid(row[0]) in selected]
    if busy:
        raise RuntimeError("Selected GPUs have existing compute processes; batch calibration requires idle GPUs. "
                           f"Reported PIDs: {', '.join(row[1] for row in busy)}")


def _lock_gpus(stack: ExitStack, selected: set[str]) -> None:
    import fcntl

    for identity in sorted(selected):
        if len(identity) != 36 or any(character not in "0123456789abcdef-" for character in identity):
            raise RuntimeError("Cannot safely identify the selected GPU UUID for calibration locking.")
        path = Path(tempfile.gettempdir()) / f"gpu-stress-tune-{os.getuid()}-{identity}.lock"
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        handle = stack.enter_context(os.fdopen(descriptor, "r+"))
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise RuntimeError(f"GPU lock is not a regular file owned by this user: {path}")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another autotuner owns GPU {identity}.") from exc


def _failure(trial_dir: Path, return_code: int, timed_out: bool) -> dict[str, Any]:
    errors = []
    for path in sorted(trial_dir.glob("failure_rank*.json")):
        with path.open(encoding="utf-8") as handle:
            errors.append(json.load(handle))
    kinds = {error.get("failure_kind") for error in errors}
    # An OOM on one rank can cause secondary collective failures on its peers.
    # A thermal/non-finite failure always takes precedence over memory backoff.
    if "thermal" in kinds:
        status = "thermal"
    elif "nonfinite" in kinds:
        status = "nonfinite"
    elif "oom" in kinds:
        status = "oom"
    elif timed_out:
        status = "timeout"
    elif "memory_preflight" in kinds:
        status = "memory_preflight"
    elif "interrupted" in kinds:
        status = "interrupted"
    else:
        status = "failed"
    return {"status": status, "return_code": return_code, "timed_out": timed_out,
            "failures": errors, "reasons": [error.get("message", "") for error in errors]
            or [f"Candidate did not complete; inspect {trial_dir / 'process.log'}"]}


def _save_trials(root: Path, result: dict[str, Any]) -> None:
    _write_json(root / "tuning_results.json", result)
    columns = ["trial", "batch_size", "status", "weighted_samples_per_second", "weighted_tflops",
               "min_rank_utilization_pct", "target_met", "peak_memory_fraction", "measured_seconds", "run_dir"]
    temporary = root / "tuning_results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["trials"])
    temporary.replace(root / "tuning_results.csv")


def run_autotune(config: StressTestConfig, tuning: AutotuneConfig) -> Path:
    """Measure a bounded search and save a recommendation; never launch a full run."""
    config.validate()
    tuning.validate(config)
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Autotuning currently requires Linux process groups; ordinary runs are unchanged.")
    if any(key in os.environ for key in ("LOCAL_RANK", "RANK", "LOCAL_WORLD_SIZE")) or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Launch autotuning once with python run.py; it manages DDP children itself. "
                           "For an external torchrun launch, pass --no-autotune.")
    packages = ["torch", "pytorch-lightning"] + (["matplotlib"] if config.generate_report else [])
    for package in packages:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"{package} is not installed in {sys.executable}. Install the project requirements first.") from exc
    root = config.run_dir.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    temperature = config.max_temperature_c if config.max_temperature_c is not None else 80.0
    memory_fraction = min(tuning.memory_fraction, config.memory_fraction or tuning.memory_fraction)
    result: dict[str, Any] = {
        "status": "running", "requested_config": config.to_dict(), "tuning": asdict(tuning),
        "effective_temperature_limit_c": temperature, "effective_memory_fraction": memory_fraction,
        "trials": [], "recommendation": None, "schema_version": 1,
        "report_status": "pending" if config.generate_report else "disabled",
        "note": "Best tested batch under current conditions; sampled utilization is not a compute-saturation guarantee.",
    }
    _save_trials(root, result)
    print(f"Autotuning: up to {tuning.max_trials} trials; {tuning.seconds:g} measured seconds each; "
          f"target {tuning.target_utilization:g}% on every GPU; allocator cap {memory_fraction:.0%}; "
          f"sampled thermal stop {temperature:g} C.", flush=True)
    try:
        discovery = root / "devices.json"
        code, timeout = _run_child(command_for_config(config) + ["--device-info-file", str(discovery),
                                  "--failure-dir", str(root)], root / "discovery.log", tuning.timeout_seconds)
        if code or timeout or not discovery.is_file():
            raise RuntimeError(f"Device inspection failed; see {root / 'discovery.log'}.")
        runtime = json.loads(discovery.read_text(encoding="utf-8"))
        result["runtime"] = runtime
        selected = {_uuid(gpu["uuid"]) for gpu in runtime["gpus"]}
        if len(selected) != runtime["world_size"] or "" in selected:
            raise RuntimeError("CUDA UUIDs do not uniquely identify every selected GPU.")
        with ExitStack() as stack:
            _lock_gpus(stack, selected)
            batch = max(1, config.batch_size // 2)
            lower = 0
            upper: int | None = None
            tested: set[int] = set()
            for number in range(1, tuning.max_trials + 1):
                _check_idle(selected)
                trial_dir = root / f"trial_{number:02d}_batch_{batch}"
                trial_dir.mkdir()
                trial_config = replace(
                    config, batch_size=batch, num_batches=max(100000, config.warmup_steps + 100000), num_epochs=1,
                    warmup_steps=max(5, config.warmup_steps), log_every_n_steps=100000,
                    log_every_n_seconds=1.0, measurement_seconds=tuning.seconds, min_measurement_windows=3,
                    memory_fraction=memory_fraction, max_temperature_c=temperature,
                    metrics_root=trial_dir, run_name="run", generate_report=False,
                )
                print(f"Trial {number}/{tuning.max_trials}: batch size {batch} per GPU", flush=True)
                started = time.monotonic()
                code, timeout = _run_child(command_for_config(trial_config) + ["--failure-dir", str(trial_dir)],
                                           trial_dir / "process.log", tuning.timeout_seconds)
                trial: dict[str, Any] = {"trial": number, "batch_size": batch, "run_dir": str(trial_config.run_dir),
                                         "wall_seconds": time.monotonic() - started}
                if code or timeout:
                    trial.update(_failure(trial_dir, code, timeout))
                else:
                    trial.update(analyze_trial(trial_config.log_path, trial_config.run_dir / "config.json",
                                               tuning.target_utilization, memory_fraction, tuning.seconds))
                result["trials"].append(trial)
                tested.add(batch)
                _save_trials(root, result)
                print(f"  {trial['status']}: throughput={trial.get('weighted_samples_per_second')}, "
                      f"minimum GPU utilization={trial.get('min_rank_utilization_pct')}", flush=True)
                if trial["status"] == "eligible":
                    lower = max(lower, batch)
                elif trial["status"] in {"oom", "memory_limit"}:
                    upper = batch if upper is None else min(upper, batch)
                else:
                    raise RuntimeError(f"Calibration stopped on {trial['status']}; see {trial_dir}. No automatic retry for this failure.")
                if upper is not None:
                    candidate = (lower + upper) // 2 if lower else max(1, batch // 2)
                elif batch < config.batch_size:
                    candidate = config.batch_size
                else:
                    candidate = min(tuning.max_batch_size, batch * 2)
                if candidate in tested or candidate < 1 or candidate > tuning.max_batch_size:
                    break
                batch = candidate
        recommendation = select_recommendation(result["trials"])
        result["recommendation"] = recommendation
        result["status"] = "completed" if recommendation is not None else "no_eligible_candidate"
        if recommendation is not None:
            recommended = replace(config, batch_size=int(recommendation["batch_size"]), run_name=default_run_name(),
                                  memory_fraction=memory_fraction,
                                  max_temperature_c=temperature, log_every_n_seconds=1.0,
                                  log_every_n_steps=100000, metrics_root=config.metrics_root.resolve())
            _write_json(root / "recommended_config.json", recommended.to_dict())
            command = shlex.join(command_for_config(recommended))
            (root / "recommended_command.txt").write_text(command + "\n", encoding="utf-8")
        _save_trials(root, result)
        if config.generate_report:
            try:
                from utils.tuning_report import generate_tuning_plot

                generate_tuning_plot(result["trials"], recommendation, root / "autotune.png",
                                     target_utilization=tuning.target_utilization, max_memory_fraction=memory_fraction)
            except Exception:
                result["report_status"] = "failed"
                if result["status"] == "completed":
                    result["status"] = "completed_report_failed"
                raise
            result["report_status"] = "completed"
            _save_trials(root, result)
        if recommendation is None:
            raise RuntimeError(f"No eligible batch found. See {root / 'tuning_results.json'}.")
        print(f"Best tested batch: {recommendation['batch_size']}; utilization target met: {recommendation['target_met']}", flush=True)
        return root
    except BaseException as exc:
        if result["status"] == "running":
            result["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "aborted"
        result["error"] = f"{type(exc).__name__}: {exc}"
        _save_trials(root, result)
        raise
