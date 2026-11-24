"""CLI entrypoint. Help and --dry-run never import PyTorch or access a GPU."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from utils.autotune import AutotuneConfig, run_autotune
from utils.config import StressTestConfig
from utils.failures import record_failure
from utils.flops import estimate_workload


def build_parser() -> argparse.ArgumentParser:
    defaults = StressTestConfig()
    parser = argparse.ArgumentParser(
        description="Train a synthetic dense network and record estimated matmul throughput and GPU telemetry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    integer_options = {
        "batch_size": "Samples per batch per device.",
        "num_batches": "Batches per epoch per device, including any warmup.",
        "num_epochs": "Training epochs.",
        "input_dim": "Input feature width.",
        "hidden_dim": "Hidden layer width.",
        "output_dim": "Output feature width.",
        "num_hidden_layers": "Exact number of hidden linear layers (at least one).",
        "num_workers": "CPU data-loader workers (ignored for on-device generation).",
        "extra_matmul_repeats": "Independent forward-only square matmuls per batch.",
        "extra_matmul_dim": "Side length of each extra square matmul.",
        "log_every_n_steps": "Measured batches per logging window; partial epoch windows are also saved.",
        "warmup_steps": "Initial batches excluded from throughput measurements, across all epochs.",
        "min_measurement_windows": "Minimum complete windows before a timed measurement may stop; batch/epoch limits still apply.",
        "devices": "Number of visible GPUs to use; -1 selects all. CPU mode requires 1.",
        "seed": "Random seed; timing is not deterministic.",
    }
    for name, help_text in integer_options.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name), help=help_text)
    parser.add_argument("--generate-on-device", dest="generate_data_on_device", action="store_true",
                        default=defaults.generate_data_on_device, help="Generate synthetic inputs on the training device.")
    parser.add_argument("--no-generate-on-device", dest="generate_data_on_device", action="store_false",
                        help="Generate inputs in the CPU data loader.")
    parser.add_argument("--accelerator", choices=("gpu", "cpu"), default=defaults.accelerator,
                        help="GPU mode requires CUDA; CPU mode must be requested explicitly.")
    parser.add_argument("--precision", choices=("auto", "32-true", "16-mixed", "bf16-mixed"),
                        default=defaults.precision, help="auto selects BF16 on compatible GPUs, otherwise FP16; CPU uses FP32.")
    parser.add_argument("--float32-matmul-precision", choices=("highest", "high", "medium"),
                        default=defaults.float32_matmul_precision, help="PyTorch internal FP32 matmul precision policy.")
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Adam learning rate.")
    parser.add_argument("--max-temperature-c", type=float, default=None,
                        help="Abort at a sampled GPU temperature threshold; requires readable, matched telemetry.")
    parser.add_argument("--memory-fraction", type=float, default=None,
                        help="Optional CUDA allocator cap as a fraction of total VRAM; also reserve free-memory headroom.")
    parser.add_argument("--measurement-seconds", type=float, default=None,
                        help="Stop after this many measured seconds (excluding warmup), or the batch/epoch limit, whichever comes first.")
    parser.add_argument("--log-every-n-seconds", type=float, default=None,
                        help="Also close measurement windows after this many seconds; checked at batch boundaries.")
    parser.add_argument("--metrics-dir", dest="metrics_root", type=Path, default=defaults.metrics_root,
                        help="Parent directory for a new, never-overwritten run folder.")
    parser.add_argument("--run-name", default=defaults.run_name, help="Run directory name (single filename component).")
    parser.add_argument("--log-filename", default=defaults.log_filename, help="CSV filename inside the run directory.")
    parser.add_argument("--no-report", dest="generate_report", action="store_false", default=defaults.generate_report,
                        help="Save CSV/metadata without generating PDF, PNG, and summary JSON.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print validated configuration and analytical size/FLOP estimates; no imports of torch, GPU access, or files written.")
    tuning = AutotuneConfig()
    parser.add_argument("--autotune", action="store_true",
                        help="Run bounded GPU calibration trials and save a batch-size recommendation; never start a full run automatically.")
    for name, kind, description in (
        ("max_batch_size", int, "Largest batch size per GPU to consider."),
        ("max_trials", int, "Maximum number of isolated candidate runs."),
        ("seconds", float, "Measured seconds per candidate, excluding warmup (3..60)."),
        ("timeout_seconds", float, "Whole-process timeout per candidate, including startup and warmup."),
        ("target_utilization", float, "Target sampled utilization on every GPU; throughput determines the shortlist."),
        ("memory_fraction", float, "Maximum VRAM fraction for calibration (0..0.95); default leaves 15% headroom."),
    ):
        parser.add_argument(f"--tune-{name.replace('_', '-')}", type=kind,
                            default=getattr(tuning, name), help=description)
    # Private subprocess control channels; these never trigger calibration by themselves.
    parser.add_argument("--device-info-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--failure-dir", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = vars(parser.parse_args())
    dry_run = args.pop("dry_run")
    autotune = args.pop("autotune")
    device_info_file = args.pop("device_info_file")
    failure_dir = args.pop("failure_dir")
    tuning_args = {name.removeprefix("tune_"): args.pop(name) for name in list(args) if name.startswith("tune_")}
    try:
        config = StressTestConfig(**args)
        tuning = AutotuneConfig(**tuning_args)
        if autotune:
            tuning.validate(config)
            if device_info_file is not None:
                raise ValueError("Device inspection and autotuning cannot be requested together.")
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    if dry_run:
        payload = {"config": config.to_dict(), "workload": estimate_workload(config)}
        if autotune:
            payload["autotune"] = asdict(tuning)
        print(json.dumps(payload, indent=2))
        return 0

    # Importing the training stack is intentionally deferred until an actual run is requested.
    try:
        if autotune:
            directory = run_autotune(config, tuning)
            print(f"Calibration complete. Results: {directory}")
            print(f"Suggested command (not executed): {directory / 'recommended_command.txt'}")
            return 0
        if device_info_file is not None:
            from utils.runner import _resolve_runtime, _write_json_atomic

            _write_json_atomic(device_info_file, _resolve_runtime(config, estimate_workload(config)))
            return 0
        from utils.runner import run_stress_test

        is_primary = run_stress_test(config, failure_dir=failure_dir)
    except KeyboardInterrupt as exc:
        _record_failure_safely(failure_dir, exc)
        print("Run interrupted. Any completed measurement windows remain in the run folder.", file=sys.stderr)
        return 130
    except Exception as exc:
        _record_failure_safely(failure_dir, exc)
        print(f"Run failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    if is_primary:
        print(f"Run complete. Metrics: {config.log_path}")
        for filename in ("config.json", "report.pdf", "summary.png", "summary.json"):
            path = config.run_dir / filename
            if path.is_file():
                print(f"Saved: {path}")
    return 0


def _record_failure_safely(directory: Path | None, exception: BaseException) -> None:
    try:
        record_failure(directory, exception)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Could not save failure details: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
