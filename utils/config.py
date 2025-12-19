from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_root() -> Path:
    # repo_root/utils/config.py -> repo_root
    return Path(__file__).resolve().parent.parent / "metrics"


def default_run_name() -> str:
    return dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")


@dataclass
class StressTestConfig:
    batch_size: int = 256
    num_batches: int = 400
    num_epochs: int = 5
    input_dim: int = 4096
    hidden_dim: int = 8192
    output_dim: int = 4096
    num_hidden_layers: int = 8  # number of hidden linear layers (>=1)
    num_workers: int = 8
    generate_data_on_device: bool = True
    extra_matmul_repeats: int = 0  # extra square matmuls per batch to increase compute
    extra_matmul_dim: int = 4096  # dimension of square matmul; large values raise FLOPS
    log_every_n_steps: int = 100
    devices: int = 1  # default to a single GPU (GPU 0)
    metrics_root: Path = field(default_factory=default_metrics_root)
    run_name: str = field(default_factory=default_run_name)
    log_filename: str = "metrics.csv"

    @property
    def dataset_size(self) -> int:
        return self.num_batches * self.batch_size

    @property
    def run_dir(self) -> Path:
        return self.metrics_root / self.run_name

    @property
    def log_path(self) -> Path:
        return self.run_dir / self.log_filename
