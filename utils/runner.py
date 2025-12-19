from __future__ import annotations

import csv
import os

import pytorch_lightning as pl
import torch
import torch.distributed as dist

from utils.config import StressTestConfig
from utils.data import build_dataloader
from utils.model import DummyLightningModel
from utils.monitor import PerformanceMonitor
from utils.create_pdf import generate_report


def run_stress_test(config: StressTestConfig) -> None:
    # Ensure a single run name across all ranks/processes.
    env_key = "GPU_STRESS_RUN_NAME"
    if env_key in os.environ:
        config.run_name = os.environ[env_key]
    else:
        os.environ[env_key] = config.run_name

    # Only rank 0 prepares the run directory and config file.
    if _is_global_zero():
        config.run_dir.mkdir(parents=True, exist_ok=True)
        config_path = config.run_dir / "config.json"
        try:
            import json

            with config_path.open("w") as f:
                json.dump(config.__dict__, f, indent=2, default=str)
        except Exception:
            pass
    # Ensure all ranks see the directory before proceeding
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    torch.set_float32_matmul_precision("high")
    precision = 16

    model = DummyLightningModel(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        num_hidden_layers=config.num_hidden_layers,
        generate_on_device=config.generate_data_on_device,
        batch_size=config.batch_size,
        extra_matmul_repeats=config.extra_matmul_repeats,
        extra_matmul_dim=config.extra_matmul_dim,
    )

    if torch.cuda.is_available():
        devices = config.devices if config.devices != -1 else torch.cuda.device_count()
        if isinstance(devices, int) and devices > torch.cuda.device_count():
            devices = torch.cuda.device_count()
        accelerator = "gpu"
        strategy = "ddp" if devices and devices > 1 else None
        active_gpu_ids = list(range(devices if isinstance(devices, int) else torch.cuda.device_count()))
        world_size = devices if isinstance(devices, int) else len(active_gpu_ids)
    else:
        devices = 1
        accelerator = "cpu"
        strategy = None
        active_gpu_ids = None
        world_size = 1

    effective_dataset_size = config.dataset_size * max(1, world_size)
    train_loader = build_dataloader(
        dataset_size=effective_dataset_size,
        batch_size=config.batch_size,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_workers=config.num_workers,
        generate_on_device=config.generate_data_on_device,
    )

    monitor = PerformanceMonitor(
        model.theoretical_flops_per_sample,
        config.batch_size,
        config.log_path,
        log_every_n_steps=config.log_every_n_steps,
        gpu_ids=active_gpu_ids,
    )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        limit_train_batches=config.num_batches,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        enable_checkpointing=False,
        logger=False,
        enable_model_summary=False,
        callbacks=[monitor],
    )

    trainer.fit(model, train_dataloaders=train_loader)

    # Merge per-rank CSVs and generate report on rank 0 only
    if _is_global_zero():
        _merge_rank_metrics(config.log_path)
        config_path = config.run_dir / "config.json"
        report_path = config.run_dir / "report.pdf"
        if config_path.exists() and config.log_path.exists():
            try:
                generate_report(config.log_path, config_path, report_path)
            except Exception:
                pass


def _merge_rank_metrics(target_path: Path) -> None:
    """Merge rank-specific CSVs into a single metrics.csv."""
    run_dir = target_path.parent
    stem = target_path.stem
    suffix = target_path.suffix
    rank_files = sorted(run_dir.glob(f"{stem}_rank*{suffix}"))

    if not rank_files:
        return

    with target_path.open("w", newline="") as out_f:
        writer = None
        for idx, rf in enumerate(rank_files):
            with rf.open() as in_f:
                reader = csv.reader(in_f)
                rows = list(reader)
                if not rows:
                    continue
                header, data_rows = rows[0], rows[1:]
                if writer is None:
                    writer = csv.writer(out_f)
                    writer.writerow(header)
                for row in data_rows:
                    writer.writerow(row)


def _is_global_zero() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True
