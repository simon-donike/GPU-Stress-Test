from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl


class DummyLightningModel(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 2,
        generate_on_device: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 1,
        extra_matmul_repeats: int = 0,
        extra_matmul_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.generate_on_device = generate_on_device
        self.batch_size = batch_size
        self.extra_matmul_repeats = extra_matmul_repeats
        self.extra_matmul_dim = extra_matmul_dim or hidden_dim
        self.compute_mat = None  # lazily created on device
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(1, num_hidden_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int):  # noqa: ANN001, D401
        if self.generate_on_device:
            # Generate synthetic data on the current device to remove CPU/dataloader bottlenecks.
            batch_size = batch[0].shape[0] if isinstance(batch, (tuple, list)) else self.batch_size
            x = torch.randn(batch_size, self.hparams.input_dim, device=self.device)
            y = torch.randn(batch_size, self.hparams.output_dim, device=self.device)
        else:
            x, y = batch
        preds = self(x)
        if self.extra_matmul_repeats > 0:
            if self.compute_mat is None or self.compute_mat.device != preds.device:
                dim = self.extra_matmul_dim
                self.compute_mat = torch.randn(dim, dim, device=preds.device)
            # Use the first dim columns to match compute matrix width
            mat_input = preds
            if preds.shape[-1] > self.compute_mat.shape[0]:
                mat_input = preds[..., : self.compute_mat.shape[0]]
            for _ in range(self.extra_matmul_repeats):
                mat_input = torch.matmul(mat_input, self.compute_mat)
            preds = mat_input
        loss = self.loss_fn(preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @property
    def theoretical_flops_per_sample(self) -> int:
        # Each linear layer costs roughly 2 * in_features * out_features per sample.
        flops = 2 * self.hparams.input_dim * self.hparams.hidden_dim
        for _ in range(max(1, self.hparams.num_hidden_layers - 1)):
            flops += 2 * self.hparams.hidden_dim * self.hparams.hidden_dim
        flops += 2 * self.hparams.hidden_dim * self.hparams.output_dim
        # Extra matmul FLOPS (square matmul: 2 * n^3 per sample is too high; approximate per batch)
        if self.extra_matmul_repeats > 0:
            flops += self.extra_matmul_repeats * 2 * (self.extra_matmul_dim**2) * max(1, self.batch_size)
        return flops
