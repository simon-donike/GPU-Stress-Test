"""Pure-Python sizing and analytical FLOP estimates for dry runs.

This module intentionally has no torch dependency.  Estimates cover dense
linear GEMMs only (FMA = 2 FLOPs); bias, activations, loss, Adam update,
communication, and data movement are excluded from FLOPs.  The memory lower
bound is FP32 parameters + gradients + two FP32 Adam moments (16 bytes/param).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _get(config: Any, name: str) -> int:
    value = config[name] if isinstance(config, Mapping) else getattr(config, name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class FlopEstimate:
    parameter_count: int
    memory_lower_bound_bytes: int
    forward_flops_per_sample: int
    training_flops_per_sample: int
    extra_flops_per_batch: int
    flops_per_batch: int


def linear_dimensions(config: Any) -> tuple[tuple[int, int], ...]:
    input_dim = _get(config, "input_dim")
    hidden_dim = _get(config, "hidden_dim")
    output_dim = _get(config, "output_dim")
    hidden_layers = _get(config, "num_hidden_layers")
    return ((input_dim, hidden_dim),) + ((hidden_dim, hidden_dim),) * (hidden_layers - 1) + ((hidden_dim, output_dim),)


def estimate_flops(config: Any, batch_size: int | None = None) -> FlopEstimate:
    """Return model parameter, memory, and dense-GEMM estimates from config."""
    dims = linear_dimensions(config)
    batch = _get(config, "batch_size") if batch_size is None else batch_size
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("batch_size must be a positive integer")
    params = sum(in_dim * out_dim + out_dim for in_dim, out_dim in dims)
    forward = sum(2 * in_dim * out_dim for in_dim, out_dim in dims)
    first, *remaining = dims
    training = 4 * first[0] * first[1] + sum(6 * in_dim * out_dim for in_dim, out_dim in remaining)
    repeats = config.get("extra_matmul_repeats", 0) if isinstance(config, Mapping) else getattr(config, "extra_matmul_repeats", 0)
    dimension = config.get("extra_matmul_dim") if isinstance(config, Mapping) else getattr(config, "extra_matmul_dim", None)
    dimension = hidden_dim_from_dims(dims) if dimension is None else dimension
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0 or isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("extra matmul settings must be a non-negative repeat count and positive dimension")
    extra = 2 * repeats * dimension**3
    return FlopEstimate(params, params * 16, forward, training, extra, batch * training + extra)


def hidden_dim_from_dims(dims: tuple[tuple[int, int], ...]) -> int:
    return dims[0][1]


def estimate_parameter_count(config: Any) -> int:
    return estimate_flops(config).parameter_count


def estimate_memory_lower_bound_bytes(config: Any) -> int:
    return estimate_flops(config).memory_lower_bound_bytes


def estimate_workload(config: Any, batch_size: int | None = None) -> dict[str, Any]:
    """Return a JSON-serializable dry-run summary for CLI/report consumers."""
    estimate = estimate_flops(config, batch_size=batch_size)
    return {
        "parameter_count": estimate.parameter_count,
        "memory_lower_bound_bytes": estimate.memory_lower_bound_bytes,
        "memory_lower_bound_gib": estimate.memory_lower_bound_bytes / (1024**3),
        "forward_flops_per_sample": estimate.forward_flops_per_sample,
        "training_flops_per_sample": estimate.training_flops_per_sample,
        "extra_flops_per_batch": estimate.extra_flops_per_batch,
        "flops_per_batch": estimate.flops_per_batch,
        "flop_convention": "Dense GEMMs only; one multiply-add (FMA) counts as 2 FLOPs.",
        "exclusions": "Bias, activation, loss, Adam update, communication, and data movement are excluded.",
    }
