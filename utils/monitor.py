"""Synchronized wall-time windows, rank-local work counts, and GPU telemetry."""
from __future__ import annotations

import csv
import datetime as dt
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist

from utils.config import StressTestConfig
from utils.failures import NonFiniteLossError, ThermalGuardError

CSV_COLUMNS = [
    "timestamp", "window", "step", "epoch", "rank", "local_rank", "world_size",
    "device_type", "device_name", "device_uuid", "batch_count", "sample_count",
    "elapsed_seconds", "window_seconds", "forward_flops", "training_flops", "extra_flops",
    "total_flops", "tflops_per_device", "tflops_total", "samples_per_second_total",
    "gpu_temp_c", "gpu_util_pct", "gpu_mem_used_gib", "gpu_mem_total_gib", "gpu_mem_used_pct",
    "gpu_power_w", "gpu_power_limit_w", "cuda_peak_allocated_gib", "cuda_peak_reserved_gib",
]


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalize_uuid(value: Any) -> str:
    # CUDA properties expose bare UUIDs; nvidia-smi prefixes physical GPUs with GPU-.
    # Do not collapse MIG identifiers onto their parent GPU.
    return str(value or "").strip().lower().removeprefix("gpu-")


def query_gpu_metrics_all(timeout_seconds: float = 2.0) -> list[dict[str, Any]]:
    """Read each field independently; unsupported readings remain unavailable."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for parts in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
        if len(parts) != 9:
            continue
        parts = [part.strip() for part in parts]
        rows.append({
            "uuid": _normalize_uuid(parts[1]),
            "temp_c": _number(parts[3]),
            "util_pct": _number(parts[4]),
            "mem_used_mib": _number(parts[5]),
            "mem_total_mib": _number(parts[6]),
            "power_w": _number(parts[7]),
            "power_limit_w": _number(parts[8]),
        })
    return rows


class PerformanceMonitor(pl.Callback):
    """One CSV writer, one row per rank/window, no extrapolation from rank zero.

    CUDA synchronization surrounds each measurement window. The per-batch finite
    loss check synchronizes too; its overhead is deliberately part of the window.
    Telemetry, metadata collectives, and CSV writes happen after its end timestamp.
    """

    def __init__(self, config: StressTestConfig, csv_path: Path | None = None) -> None:
        super().__init__()
        self.config = config
        self._explicit_csv_path = csv_path
        self._csv_path: Path | None = None
        self._device = torch.device("cpu")
        self._identity: dict[str, str] = {}
        self._flop_counts: dict[str, int] = {}
        self._warmup_seen = 0
        self._window_number = 0
        self._measured_seconds = 0.0
        self._last_thermal_check = 0.0
        self._reset_window()

    def _reset_window(self) -> None:
        self._window_start: float | None = None
        self._window_batches = 0
        self._window_samples = 0

    @property
    def _cuda(self) -> bool:
        return self._device.type == "cuda"

    @staticmethod
    def _distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _sync(self) -> None:
        if self._cuda:
            torch.cuda.synchronize(self._device)

    def _gather(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if not self._distributed():
            return [payload]
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, payload)
        return gathered

    @staticmethod
    def _raise_writer_error(trainer: pl.Trainer, error: str | None) -> None:
        error = trainer.strategy.broadcast(error, src=0)
        if error:
            raise RuntimeError(error)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._device = pl_module.device
        # Lifecycle callback has now broadcast the rank-zero output location.
        self._csv_path = Path(self._explicit_csv_path or self.config.log_path)
        if self._cuda:
            props = torch.cuda.get_device_properties(self._device)
            self._identity = {"device_type": "CUDA", "device_name": props.name,
                              "device_uuid": _normalize_uuid(getattr(props, "uuid", ""))}
        else:
            self._identity = {"device_type": "CPU", "device_name": "CPU", "device_uuid": ""}
        for name in ("forward_flops_per_sample", "training_flops_per_sample", "extra_flops_per_batch"):
            value = getattr(pl_module, name, None)
            minimum = 0 if name == "extra_flops_per_batch" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise RuntimeError(f"Model {name} must be an integer >= {minimum}; refusing to guess FLOPs.")
            self._flop_counts[name] = value
        error = None
        if trainer.is_global_zero:
            try:
                with self._csv_path.open("x", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=CSV_COLUMNS).writeheader()
            except (OSError, csv.Error, TypeError, ValueError) as exc:
                error = f"Cannot create metrics CSV {self._csv_path}: {exc}"
        self._raise_writer_error(trainer, error)
        self._check_thermal_before_work("preflight")
        self._last_thermal_check = time.perf_counter()

    def _telemetry(self) -> dict[str, Any]:
        if not self._cuda or not self._identity["device_uuid"]:
            return {}
        matches = [row for row in query_gpu_metrics_all() if row["uuid"] == self._identity["device_uuid"]]
        return matches[0] if len(matches) == 1 else {}

    def _thermal_error(self, telemetry: dict[str, Any]) -> str | None:
        limit = self.config.max_temperature_c
        if limit is None:
            return None
        temperature = telemetry.get("temp_c")
        if temperature is None:
            return "Thermal guard requires a readable temperature for the CUDA device UUID; check nvidia-smi and device mapping."
        if temperature >= limit:
            return f"Thermal guard reached {temperature:.1f} C (limit {limit:.1f} C); stop and let the device cool."
        return None

    def _check_thermal_before_work(self, context: str) -> None:
        if self.config.max_temperature_c is None:
            return
        errors = self._gather({"error": self._thermal_error(self._telemetry())})
        for rank, item in enumerate(errors):
            if item["error"]:
                raise ThermalGuardError(f"Rank {rank}, {context}: {item['error']}")

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                             batch: Any, batch_idx: int) -> None:
        if self._warmup_seen < self.config.warmup_steps:
            self._warmup_seen += 1
            return
        if self._window_start is None:
            self._sync()
            if self._cuda:
                torch.cuda.reset_peak_memory_stats(self._device)
            self._window_start = time.perf_counter()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if torch.is_tensor(loss):
            finite = bool(torch.isfinite(loss.detach()).all().item())
        else:
            finite = _number(loss) is not None
        current_seconds = time.perf_counter() - self._window_start if self._window_start is not None else 0.0
        stop = (self._window_start is not None and self.config.measurement_seconds is not None
                and self._window_number + 1 >= self.config.min_measurement_windows
                and self._measured_seconds + current_seconds >= self.config.measurement_seconds)
        timed_flush = (self._window_start is not None and self.config.log_every_n_seconds is not None
                       and current_seconds >= self.config.log_every_n_seconds)
        warmup_thermal = (self._window_start is None and self.config.max_temperature_c is not None
                          and (self._warmup_seen % self.config.log_every_n_steps == 0
                               or self._warmup_seen == self.config.warmup_steps
                               or time.perf_counter() - self._last_thermal_check >= 1.0))
        flags = [int(not finite), int(timed_flush), int(stop), int(warmup_thermal)]
        if self._distributed():
            # Every rank must make the same flush/stop decision. Piggyback on
            # the finite-loss collective rather than add another per-step sync.
            shared_flags = torch.tensor(flags, device=self._device)
            dist.all_reduce(shared_flags, op=dist.ReduceOp.MAX)
            flags = shared_flags.tolist()
        if flags[0]:
            raise NonFiniteLossError("Missing or non-finite loss on at least one rank. Reduce the learning rate or use BF16/FP32 precision.")
        if self._window_start is None:
            # Sample through warmup as well, including its final batch.
            if flags[3]:
                self._check_thermal_before_work("warmup")
                self._last_thermal_check = time.perf_counter()
            return
        item = batch[0] if isinstance(batch, (tuple, list)) else batch
        if not torch.is_tensor(item) or item.ndim < 1 or item.shape[0] < 1:
            raise RuntimeError("Cannot determine a positive actual batch size; refusing to guess FLOPs.")
        self._window_batches += 1
        self._window_samples += int(item.shape[0])
        if self._window_batches >= self.config.log_every_n_steps or trainer.is_last_batch or flags[1] or flags[2]:
            self._flush(trainer)
        if flags[2]:
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._window_batches:
            self._flush(trainer)

    def _flush(self, trainer: pl.Trainer) -> None:
        if self._window_start is None or not self._window_batches:
            return
        self._sync()
        elapsed = time.perf_counter() - self._window_start
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Measured window duration is invalid; refusing to report throughput.")
        # Everything below is excluded from the measured window.
        telemetry = self._telemetry()
        training_flops = self._window_samples * self._flop_counts["training_flops_per_sample"]
        extra_flops = self._window_batches * self._flop_counts["extra_flops_per_batch"]
        used, capacity = telemetry.get("mem_used_mib"), telemetry.get("mem_total_mib")
        payload = {
            **self._identity,
            "rank": trainer.global_rank,
            "local_rank": trainer.local_rank,
            "batch_count": self._window_batches,
            "sample_count": self._window_samples,
            "elapsed_seconds": elapsed,
            "forward_flops": self._window_samples * self._flop_counts["forward_flops_per_sample"],
            "training_flops": training_flops,
            "extra_flops": extra_flops,
            "total_flops": training_flops + extra_flops,
            "gpu_temp_c": telemetry.get("temp_c"),
            "gpu_util_pct": telemetry.get("util_pct"),
            "gpu_mem_used_gib": used / 1024 if used is not None else None,
            "gpu_mem_total_gib": capacity / 1024 if capacity is not None else None,
            "gpu_mem_used_pct": used / capacity * 100 if used is not None and capacity else None,
            "gpu_power_w": telemetry.get("power_w"),
            "gpu_power_limit_w": telemetry.get("power_limit_w"),
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(self._device) / 1024**3 if self._cuda else None,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(self._device) / 1024**3 if self._cuda else None,
            "thermal_error": self._thermal_error(telemetry),
        }
        payloads = self._gather(payload)
        duration = max(item["elapsed_seconds"] for item in payloads)
        global_flops = sum(item["total_flops"] for item in payloads)
        global_samples = sum(item["sample_count"] for item in payloads)
        shared = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "window": self._window_number,
            "step": trainer.global_step,
            "epoch": trainer.current_epoch,
            "world_size": trainer.world_size,
            "window_seconds": duration,
            "tflops_total": global_flops / duration / 1e12,
            "samples_per_second_total": global_samples / duration,
        }
        error = None
        if trainer.is_global_zero:
            try:
                assert self._csv_path is not None
                with self._csv_path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                    for item in sorted(payloads, key=lambda value: value["rank"]):
                        row = {key: value for key, value in item.items() if key != "thermal_error"}
                        row.update(shared)
                        row["tflops_per_device"] = item["total_flops"] / item["elapsed_seconds"] / 1e12
                        writer.writerow(row)
                    handle.flush()
            except (OSError, csv.Error, TypeError, ValueError) as exc:
                error = f"Cannot append metrics CSV {self._csv_path}: {exc}"
        # All ranks see filesystem failures before entering another training collective.
        self._raise_writer_error(trainer, error)
        self._window_number += 1
        self._measured_seconds += duration
        self._reset_window()
        for item in payloads:
            if item["thermal_error"]:
                raise ThermalGuardError(f"Rank {item['rank']}: {item['thermal_error']}")

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        # Completed windows are closed on disk; incomplete work must never inflate their counts.
        self._reset_window()
