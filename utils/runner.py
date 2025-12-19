from __future__ import annotations

import pytorch_lightning as pl
import torch

from utils.config import StressTestConfig
from utils.data import build_dataloader
from utils.model import DummyLightningModel
from utils.monitor import PerformanceMonitor
from utils.create_pdf import generate_report


def run_stress_test(config: StressTestConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    # Save config for reproducibility
    config_path = config.run_dir / "config.json"
    try:
        import json

        with config_path.open("w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)
    except Exception:
        pass
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

    train_loader = build_dataloader(
        dataset_size=config.dataset_size,
        batch_size=config.batch_size,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        num_workers=config.num_workers,
        generate_on_device=config.generate_data_on_device,
    )
    if torch.cuda.is_available():
        devices = config.devices if config.devices != -1 else torch.cuda.device_count()
        if isinstance(devices, int) and devices > torch.cuda.device_count():
            devices = torch.cuda.device_count()
        accelerator = "gpu"
        strategy = "ddp" if devices and devices > 1 else None
        active_gpu_ids = list(range(devices if isinstance(devices, int) else torch.cuda.device_count()))
    else:
        devices = 1
        accelerator = "cpu"
        strategy = None
        active_gpu_ids = None

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

    # Generate PDF report
    config_path = config.run_dir / "config.json"
    report_path = config.run_dir / "report.pdf"
    if config_path.exists() and config.log_path.exists():
        try:
            generate_report(config.log_path, config_path, report_path)
        except Exception:
            pass
