from __future__ import annotations

import csv
import datetime as dt
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import torch


def query_gpu_metrics_all(selected_ids: Optional[List[int]] = None) -> List[Dict[str, Optional[float]]]:
    """
    Returns GPU metrics for selected GPUs (or all visible) using nvidia-smi.
    Falls back to empty list when unavailable or on CPU.
    """
    if not torch.cuda.is_available():
        return []

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        rows = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpu_id_val = int(parts[0])
            if selected_ids is not None and gpu_id_val not in selected_ids:
                continue
            rows.append(
                {
                    "gpu_id": gpu_id_val,
                    "name": parts[1],
                    "temp_c": float(parts[2]),
                    "util_pct": float(parts[3]),
                    "mem_used_mb": float(parts[4]),
                    "mem_total_mb": float(parts[5]),
                    "power_w": float(parts[6]),
                    "power_limit_w": float(parts[7]),
                }
            )
        return rows
    except Exception:
        return []


class PerformanceMonitor(pl.Callback):
    def __init__(
        self,
        flops_per_sample: int,
        batch_size: int,
        csv_path: Path,
        log_every_n_steps: int,
        gpu_ids: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.flops_per_batch = flops_per_sample * batch_size
        self.base_csv_path = csv_path
        self.rank_csv_path: Optional[Path] = None
        self._start_time: Optional[dt.datetime] = None
        self.gpu_index: Optional[int] = None
        self.log_every_n_steps = log_every_n_steps
        self.gpu_ids = gpu_ids

    def _prepare_rank_file(self, trainer) -> None:
        if self.rank_csv_path is not None:
            return
        rank = self._get_rank(trainer)
        stem = self.base_csv_path.stem
        suffix = self.base_csv_path.suffix
        self.rank_csv_path = self.base_csv_path.with_name(f"{stem}_rank{rank}{suffix}")
        if not self.rank_csv_path.exists():
            self._write_header(self.rank_csv_path)

    def _write_header(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "gpu_id",
                    "gpu_name",
                    "step",
                    "world_size",
                    "tflops_total",
                    "tflops_per_gpu",
                    "gpu_temp_c",
                    "gpu_util_pct",
                    "gpu_mem_used_gb",
                    "gpu_mem_used_pct",
                    "gpu_power_w",
                ]
            )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):  # noqa: D401, ANN001
        # torch.cuda.synchronize ensures the clock is accurate when timing GPU kernels.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = dt.datetime.utcnow()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, dataloader_idx: int = 0):  # noqa: D401, ANN001
        if self._start_time is None:
            return
        if not self._is_global_zero(trainer):
            return
        self._prepare_rank_file(trainer)

        if torch.cuda.is_available() and self.gpu_index is None:
            # Capture the active GPU index once training starts.
            self.gpu_index = torch.cuda.current_device()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (dt.datetime.utcnow() - self._start_time).total_seconds()
        self._start_time = None

        world_size = getattr(trainer, "world_size", None) or getattr(getattr(trainer, "strategy", None), "world_size", None) or 1
        tflops_total = (self.flops_per_batch * world_size / elapsed) / 1e12 if elapsed > 0 else 0.0
        tflops_per_gpu = tflops_total / world_size if world_size > 0 else tflops_total
        metrics_list = query_gpu_metrics_all(self.gpu_ids)

        # Skip writing unless we're on a log step (always log step 0).
        if trainer.global_step != 0 and (trainer.global_step % self.log_every_n_steps) != 0:
            return

        with self.rank_csv_path.open("a", newline="") as f:  # type: ignore[arg-type]
            writer = csv.writer(f)
            now = dt.datetime.utcnow().isoformat()
            if not metrics_list:
                metrics_list = [
                    {
                        "gpu_id": "n/a",
                        "name": "cpu",
                        "temp_c": None,
                        "util_pct": None,
                        "mem_used_mb": None,
                        "mem_total_mb": None,
                        "power_w": None,
                        "power_limit_w": None,
                    }
                ]
            for metrics in metrics_list:
                mem_used_gb = (metrics["mem_used_mb"] / 1024.0) if metrics.get("mem_used_mb") is not None else None
                mem_pct = (
                    (metrics["mem_used_mb"] / metrics["mem_total_mb"] * 100.0)
                    if metrics.get("mem_used_mb") is not None and metrics.get("mem_total_mb")
                    else None
                )
                writer.writerow(
                    [
                        now,
                        metrics.get("gpu_id", "n/a"),
                        metrics.get("name", "n/a"),
                        trainer.global_step,
                        world_size,
                        round(tflops_total, 3),
                        round(tflops_per_gpu, 3),
                        round(metrics["temp_c"], 1) if metrics.get("temp_c") is not None else "n/a",
                        round(metrics["util_pct"], 1) if metrics.get("util_pct") is not None else "n/a",
                        round(mem_used_gb, 3) if mem_used_gb is not None else "n/a",
                        round(mem_pct, 1) if mem_pct is not None else "n/a",
                        round(metrics["power_w"], 1) if metrics.get("power_w") is not None else "n/a",
                    ]
                )

    @staticmethod
    def _is_global_zero(trainer) -> bool:
        if hasattr(trainer, "is_global_zero"):
            return bool(trainer.is_global_zero)
        if hasattr(trainer, "global_rank"):
            return int(trainer.global_rank) == 0
        try:
            import torch.distributed as dist

            return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            return True

    @staticmethod
    def _get_rank(trainer) -> int:
        if hasattr(trainer, "global_rank"):
            return int(trainer.global_rank)
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass
        return int(os.environ.get("RANK", 0))
