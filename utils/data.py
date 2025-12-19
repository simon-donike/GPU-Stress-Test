from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class RandomDataset(Dataset):
    def __init__(self, length: int, input_dim: int, output_dim: int) -> None:
        self.length = length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        x = torch.randn(self.input_dim)
        y = torch.randn(self.output_dim)
        return x, y


class DummyBatchDataset(Dataset):
    """Lightweight dataset used when data is generated on-device."""

    def __init__(self, length: int) -> None:
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):  # noqa: D401, ANN001
        return 0


def build_dataloader(
    dataset_size: int,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    num_workers: int = 4,
    generate_on_device: bool = False,
) -> DataLoader:
    dataset = DummyBatchDataset(dataset_size) if generate_on_device else RandomDataset(dataset_size, input_dim, output_dim)
    pin_mem = torch.cuda.is_available() and not generate_on_device
    workers = 0 if generate_on_device else num_workers
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": pin_mem,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
