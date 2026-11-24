"""Validated, dependency-free configuration shared by the CLI and runner."""
from __future__ import annotations

import datetime as dt
import math
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def default_metrics_root() -> Path:
    return Path(__file__).resolve().parent.parent / "metrics"


def default_run_name() -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _validate_component(value: str, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"{name} must be a single filename component (letters, digits, '.', '_', '-').")


@dataclass
class StressTestConfig:
    # A moderate starting workload. No configuration guarantees a fit in available VRAM.
    batch_size: int = 64
    num_batches: int = 100
    num_epochs: int = 1
    input_dim: int = 1024
    hidden_dim: int = 2048
    output_dim: int = 1024
    num_hidden_layers: int = 4
    num_workers: int = 0
    generate_data_on_device: bool = True
    extra_matmul_repeats: int = 0
    extra_matmul_dim: int = 2048
    log_every_n_steps: int = 10
    warmup_steps: int = 5
    devices: int = 1
    accelerator: str = "gpu"
    precision: str = "auto"
    float32_matmul_precision: str = "highest"
    learning_rate: float = 1e-3
    seed: int = 42
    max_temperature_c: float | None = None
    memory_fraction: float | None = None
    measurement_seconds: float | None = None
    min_measurement_windows: int = 1
    log_every_n_seconds: float | None = None
    generate_report: bool = True
    metrics_root: Path = field(default_factory=default_metrics_root)
    run_name: str = field(default_factory=default_run_name)
    log_filename: str = "metrics.csv"

    def __post_init__(self) -> None:
        self.metrics_root = Path(self.metrics_root).expanduser()
        self.validate()

    def validate(self) -> None:
        for name in ("batch_size", "num_batches", "num_epochs", "input_dim", "hidden_dim", "output_dim",
                     "num_hidden_layers", "extra_matmul_dim", "log_every_n_steps", "min_measurement_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("num_workers", "extra_matmul_repeats", "warmup_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32.")
        if self.warmup_steps >= self.num_batches * self.num_epochs:
            raise ValueError("warmup_steps must be less than total batches (num_batches * num_epochs).")
        if isinstance(self.devices, bool) or not isinstance(self.devices, int) or (self.devices != -1 and self.devices < 1):
            raise ValueError("devices must be a positive integer or -1 for all visible GPUs.")
        if self.accelerator not in {"gpu", "cpu"}:
            raise ValueError("accelerator must be 'gpu' or 'cpu'.")
        if self.accelerator == "cpu" and self.devices != 1:
            raise ValueError("CPU mode supports exactly one device.")
        if self.precision not in {"auto", "32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError("precision must be auto, 32-true, 16-mixed, or bf16-mixed.")
        if self.accelerator == "cpu" and self.precision not in {"auto", "32-true"}:
            raise ValueError("CPU mode supports only auto or 32-true precision.")
        if self.float32_matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("float32_matmul_precision must be highest, high, or medium.")
        for name in ("generate_data_on_device", "generate_report"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if (isinstance(self.learning_rate, bool) or not isinstance(self.learning_rate, (int, float))
                or not math.isfinite(self.learning_rate) or self.learning_rate <= 0):
            raise ValueError("learning_rate must be finite and positive.")
        if self.max_temperature_c is not None:
            if (isinstance(self.max_temperature_c, bool) or not isinstance(self.max_temperature_c, (int, float))
                    or not math.isfinite(self.max_temperature_c) or not 0 < self.max_temperature_c <= 150):
                raise ValueError("max_temperature_c must be finite and between 0 and 150 Celsius.")
            if self.accelerator != "gpu":
                raise ValueError("max_temperature_c requires GPU mode.")
        for name in ("memory_fraction", "measurement_seconds", "log_every_n_seconds"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when supplied.")
        if self.memory_fraction is not None:
            if self.memory_fraction > 1 or self.accelerator != "gpu":
                raise ValueError("memory_fraction requires GPU mode and must be <= 1.")
        _validate_component(self.run_name, "run_name")
        _validate_component(self.log_filename, "log_filename")
        if not self.log_filename.endswith(".csv"):
            raise ValueError("log_filename must end in .csv.")

    @property
    def dataset_size(self) -> int:
        """Samples per rank and epoch; the runner scales by the DDP world size."""
        return self.num_batches * self.batch_size

    @property
    def run_dir(self) -> Path:
        return self.metrics_root / self.run_name

    @property
    def log_path(self) -> Path:
        return self.run_dir / self.log_filename

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["metrics_root"] = str(self.metrics_root)
        return values
