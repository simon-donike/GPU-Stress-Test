from __future__ import annotations

import math

import torch
import torch.nn as nn
import pytorch_lightning as pl


class DummyLightningModel(pl.LightningModule):
    """MLP stress workload with explicit dense-GEMM FLOP accounting."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int,
        num_hidden_layers: int = 2, generate_on_device: bool = False,
        learning_rate: float = 1e-3, batch_size: int = 1,
        extra_matmul_repeats: int = 0, extra_matmul_dim: int | None = None,
    ) -> None:
        super().__init__()
        values = {"input_dim": input_dim, "hidden_dim": hidden_dim,
                  "output_dim": output_dim, "num_hidden_layers": num_hidden_layers,
                  "batch_size": batch_size, "extra_matmul_repeats": extra_matmul_repeats}
        for name, value in values.items():
            minimum = 0 if name == "extra_matmul_repeats" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if extra_matmul_dim is not None and (isinstance(extra_matmul_dim, bool) or not isinstance(extra_matmul_dim, int) or extra_matmul_dim <= 0):
            raise ValueError("extra_matmul_dim must be a positive integer")
        if not isinstance(generate_on_device, bool):
            raise ValueError("generate_on_device must be a boolean")
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a finite positive number")
        self.save_hyperparameters()
        self.generate_on_device = generate_on_device
        self.batch_size = batch_size
        self.extra_matmul_repeats = extra_matmul_repeats
        self.extra_matmul_dim = extra_matmul_dim or hidden_dim
        # Kept as a plain tensor so DDP does not broadcast a potentially huge
        # auxiliary operand. It is recreated when device/dtype changes.
        self.compute_mat: torch.Tensor | None = None

        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _run_extra_matmuls(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.extra_matmul_repeats <= 0:
            return
        if self.compute_mat is None or self.compute_mat.device != device or self.compute_mat.dtype != dtype:
            with torch.no_grad():
                mat = torch.randn(self.extra_matmul_dim, self.extra_matmul_dim, device=device, dtype=dtype)
                self.compute_mat = mat.tanh() / math.sqrt(self.extra_matmul_dim)
        with torch.no_grad():
            for _ in range(self.extra_matmul_repeats):
                torch.matmul(self.compute_mat, self.compute_mat)

    def training_step(self, batch, batch_idx: int):  # noqa: ANN001, D401
        if self.generate_on_device:
            if isinstance(batch, (tuple, list)):
                actual_batch_size = batch[0].shape[0]
            elif hasattr(batch, "shape") and batch.ndim > 0:
                actual_batch_size = batch.shape[0]
            else:
                actual_batch_size = self.batch_size
            x = torch.randn(actual_batch_size, self.hparams.input_dim, device=self.device)
            y = torch.randn(actual_batch_size, self.hparams.output_dim, device=self.device)
        else:
            x, y = batch
        preds = self(x)
        self._run_extra_matmuls(preds.device, preds.dtype)
        return {"loss": self.loss_fn(preds, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def teardown(self, stage: str) -> None:
        # The auxiliary tensor is deliberately not a registered buffer, so
        # Lightning's move back to CPU cannot release it on our behalf.
        self.compute_mat = None

    @property
    def _linear_dimensions(self) -> tuple[tuple[int, int], ...]:
        return tuple((layer.in_features, layer.out_features) for layer in self.net if isinstance(layer, nn.Linear))

    @property
    def forward_flops_per_sample(self) -> int:
        """Forward dense GEMM FLOPs per sample; each FMA counts as two FLOPs."""
        return sum(2 * in_dim * out_dim for in_dim, out_dim in self._linear_dimensions)

    @property
    def training_flops_per_sample(self) -> int:
        """Training dense GEMM FLOPs; input gradients are omitted for the first layer."""
        first, *remaining = self._linear_dimensions
        return 4 * first[0] * first[1] + sum(6 * in_dim * out_dim for in_dim, out_dim in remaining)

    @property
    def extra_flops_per_batch(self) -> int:
        return 2 * self.extra_matmul_repeats * self.extra_matmul_dim**3

    def flops_for_batch(self, batch_size: int) -> int:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        return batch_size * self.training_flops_per_sample + self.extra_flops_per_batch

    @property
    def theoretical_flops_per_sample(self) -> int:
        """Compatibility alias for the training estimate."""
        return self.training_flops_per_sample
