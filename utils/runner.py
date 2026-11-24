"""Training orchestration, preflight validation, and durable run metadata."""
from __future__ import annotations

import datetime as dt
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.strategies import DDPStrategy

from utils.config import StressTestConfig
from utils.data import build_dataloader
from utils.flops import estimate_workload
from utils.failures import InsufficientMemoryError, failure_kind, record_failure
from utils.model import DummyLightningModel
from utils.monitor import PerformanceMonitor


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _resolve_runtime(config: StressTestConfig, estimate: dict[str, Any]) -> dict[str, Any]:
    """Inspect requested devices before allocating model weights; never launch kernels."""
    devices = 1
    gpu_info: list[dict[str, Any]] = []
    if config.accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and check the driver, "
                               "or explicitly select --accelerator cpu.")
        visible = torch.cuda.device_count()
        devices = visible if config.devices == -1 else config.devices
        if devices > visible:
            raise ValueError(f"Requested {devices} GPUs, but only {visible} are visible. Check CUDA_VISIBLE_DEVICES.")
        for index in range(devices):
            props = torch.cuda.get_device_properties(index)
            total = int(props.total_memory)
            lower_bound = estimate["memory_lower_bound_bytes"]
            if lower_bound >= total:
                raise ValueError(f"Model state alone requires at least {lower_bound / 1024**3:.2f} GiB on GPU {index}, "
                                 f"which has {total / 1024**3:.2f} GiB. Reduce layer count or dimensions.")
            if lower_bound > total * 0.75:
                warnings.warn(f"GPU {index}: model state alone exceeds 75% of total VRAM; activations, extra "
                              "matmuls, DDP buffers, and other processes need additional memory.", stacklevel=2)
            gpu_info.append({
                "logical_index": index,
                "name": props.name,
                "uuid": str(getattr(props, "uuid", "")),
                "total_memory_gib": total / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
                "sm_count": props.multi_processor_count,
            })
        native_bf16 = all(int(gpu["compute_capability"].split(".")[0]) >= 8 for gpu in gpu_info)
        if config.precision == "bf16-mixed" and not native_bf16:
            raise ValueError("bf16-mixed requires native BF16 support on every selected GPU (compute capability >= 8).")
        precision = ("bf16-mixed" if native_bf16 else "16-mixed") if config.precision == "auto" else config.precision
    else:
        precision = "32-true"

    # Only single-node DDP is supported; do not silently undercount an external launch.
    external_world = int(os.environ.get("WORLD_SIZE", "1"))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", str(devices)))
    if external_world not in (1, devices) or local_world != devices or int(os.environ.get("GROUP_RANK", "0")) != 0:
        raise ValueError("Only single-node execution is supported. With torchrun, --devices must match --nproc-per-node.")
    return {
        "accelerator": config.accelerator,
        "devices": devices,
        "world_size": devices,
        "precision": precision,
        "float32_matmul_precision": config.float32_matmul_precision,
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "lightning_version": pl.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpu_info,
        "timing": "Synchronized per-rank monotonic wall windows; global denominator is the slowest rank window.",
        "telemetry_scope": "Whole matched GPU; not isolated to this process. Allocator peaks are per process.",
    }


class _RunLifecycle(pl.Callback):
    """Create outputs after rank initialization, with one writer and coherent errors."""

    def __init__(self, config: StressTestConfig, runtime: dict[str, Any], estimate: dict[str, Any],
                 failure_dir: Path | None = None) -> None:
        self.config = config
        self.runtime = runtime
        self.estimate = estimate
        self.metadata: dict[str, Any] = {}
        self.created = False
        self.memory_budget: dict[str, Any] | None = None
        self.failure_dir = failure_dir

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        if self.config.memory_fraction is None or stage != "fit":
            return
        device = trainer.strategy.root_device
        error: str | None = None
        try:
            free, total = torch.cuda.mem_get_info(device)
            # Headroom covers allocations outside the caching allocator, such as NCCL.
            reserve = max(1024**3, total * (1 - self.config.memory_fraction))
            budget = min(total * self.config.memory_fraction, free - reserve)
            if budget <= self.estimate["memory_lower_bound_bytes"]:
                error = ("Insufficient free VRAM for model state plus configured memory headroom. "
                         "Reduce model dimensions or release other GPU workloads.")
        except (RuntimeError, OSError, ValueError) as exc:
            error = f"Unable to inspect free GPU memory before model setup: {exc}"
        errors = _gather_errors(trainer, error)
        if errors:
            raise InsufficientMemoryError("; ".join(f"rank {rank}: {message}" for rank, message in errors))
        try:
            torch.cuda.set_per_process_memory_fraction(float(budget / total), device)
        except (RuntimeError, OSError, ValueError) as exc:
            error = f"Unable to apply the per-process memory cap before training: {exc}"
        errors = _gather_errors(trainer, error)
        if errors:
            raise InsufficientMemoryError("; ".join(f"rank {rank}: {message}" for rank, message in errors))
        self.memory_budget = {
            "rank": trainer.global_rank, "requested_fraction": self.config.memory_fraction,
            "effective_fraction": budget / total, "allocator_budget_bytes": int(budget),
            "free_bytes_before_model": free, "total_bytes": total, "headroom_bytes": int(reserve),
        }

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.world_size != self.runtime["devices"]:
            raise RuntimeError("Actual DDP world size differs from requested devices; refusing incorrect sample/FLOP accounting.")
        # Lightning's subprocess launcher and torchrun can construct separate default names.
        location = trainer.strategy.broadcast((self.config.run_name, str(self.config.metrics_root)), src=0)
        self.config.run_name, root = location
        self.config.metrics_root = Path(root)
        self.config.validate()
        if self.config.memory_fraction is not None:
            budgets: list[Any] = [self.memory_budget]
            if dist.is_available() and dist.is_initialized():
                budgets = [None] * trainer.world_size
                dist.all_gather_object(budgets, self.memory_budget)
            self.runtime["memory_limits"] = budgets
        error = None
        if trainer.is_global_zero:
            try:
                self.config.metrics_root.mkdir(parents=True, exist_ok=True)
                # Never append to an older run, including when --run-name is explicit.
                self.config.run_dir.mkdir(exist_ok=False)
                self.created = True
                self.metadata = {
                    **self.config.to_dict(),
                    "schema_version": 2,
                    "status": "running",
                    "started_at": _utc_now(),
                    "runtime": self.runtime,
                    "flop_accounting": self.estimate,
                    "report_status": "pending" if self.config.generate_report else "disabled",
                }
                self.save()
            except (OSError, ValueError, TypeError) as exc:
                error = f"Cannot initialize run folder {self.config.run_dir}: {exc}"
        error = trainer.strategy.broadcast(error, src=0)
        if error:
            raise RuntimeError(error)

    def save(self) -> None:
        if self.created and self.metadata:
            _write_json_atomic(self.config.run_dir / "config.json", self.metadata)

    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException) -> None:
        failure_dir = self.failure_dir or (self.config.run_dir if self.created else None)
        record_failure(failure_dir, exception, rank=trainer.global_rank)
        if trainer.is_global_zero and self.created:
            self.metadata.update(status="interrupted" if isinstance(exception, KeyboardInterrupt) else "failed",
                                 finished_at=_utc_now(), error=f"{type(exception).__name__}: {exception}",
                                 failure_kind=failure_kind(exception))
            try:
                self.save()
            except OSError as exc:
                warnings.warn(f"Could not save failure status: {exc}", stacklevel=2)


def _gather_errors(trainer: pl.Trainer, error: str | None) -> list[tuple[int, str]]:
    """Collect setup errors before any rank proceeds to allocator/model work."""
    if dist.is_available() and dist.is_initialized():
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, error)
        return [(rank, message) for rank, message in enumerate(gathered) if message]
    return [(0, error)] if error else []


def run_stress_test(config: StressTestConfig, failure_dir: Path | None = None) -> bool:
    """Execute a requested run. Return whether this process owns the output artifacts."""
    config.validate()
    estimate = estimate_workload(config)
    runtime = _resolve_runtime(config, estimate)
    pl.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    model = DummyLightningModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_hidden_layers=config.num_hidden_layers,
        generate_on_device=config.generate_data_on_device,
        batch_size=config.batch_size,
        extra_matmul_repeats=config.extra_matmul_repeats,
        extra_matmul_dim=config.extra_matmul_dim,
        learning_rate=config.learning_rate,
    )
    # The estimator powers dry-run and preflight; assert the instantiated workload agrees.
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != estimate["parameter_count"] or model.flops_for_batch(config.batch_size) != estimate["flops_per_batch"]:
        raise RuntimeError("Model and analytical workload estimate disagree; refusing to record misleading FLOPS.")

    train_loader = build_dataloader(
        dataset_size=config.dataset_size * runtime["devices"],
        batch_size=config.batch_size,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_workers=config.num_workers,
        generate_on_device=config.generate_data_on_device,
        pin_memory=config.accelerator == "gpu" and not config.generate_data_on_device,
        distributed_replicas=runtime["devices"],
    )
    lifecycle = _RunLifecycle(config, runtime, estimate, failure_dir=failure_dir)
    monitor = PerformanceMonitor(config)
    # Extra compute operands are local and do not need a large broadcast every forward.
    strategy = DDPStrategy(find_unused_parameters=False, broadcast_buffers=False) if runtime["devices"] > 1 else "auto"
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        limit_train_batches=config.num_batches,
        accelerator=config.accelerator,
        devices=runtime["devices"],
        num_nodes=1,
        strategy=strategy,
        precision=runtime["precision"],
        accumulate_grad_batches=1,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[lifecycle, monitor],
    )
    trainer.fit(model, train_dataloaders=train_loader)
    if trainer.interrupted:
        raise KeyboardInterrupt
    if not trainer.is_global_zero:
        return False

    lifecycle.metadata.update(status="completed", finished_at=_utc_now())
    lifecycle.save()
    if config.generate_report:
        try:
            # Keep matplotlib out of training workers and the --no-report path.
            from utils.create_pdf import generate_report

            generate_report(config.log_path, config.run_dir / "config.json", config.run_dir / "report.pdf")
        except Exception as exc:
            lifecycle.metadata.update(report_status="failed", report_error=f"{type(exc).__name__}: {exc}")
            lifecycle.save()
            raise RuntimeError(f"Training completed, but reporting failed: {exc}. CSV and config are saved in "
                               f"{config.run_dir}; regenerate the report offline.") from exc
        lifecycle.metadata["report_status"] = "completed"
        lifecycle.save()
    return True
