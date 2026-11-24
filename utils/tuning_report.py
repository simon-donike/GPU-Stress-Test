"""Offline visualisation for batch-size autotuning results.

This module only consumes already saved trial dictionaries.  It deliberately
does not import the training or GPU stack, so report generation remains safe
on a machine without CUDA.
"""
from __future__ import annotations

import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_METRICS = (
    ("weighted_samples_per_second", "Weighted samples / second", "#2563eb"),
    ("min_rank_utilization_pct", "Minimum rank utilization (%)", "#059669"),
    ("peak_memory_fraction", "Memory fraction (allocator peak and sampled board use)", "#d97706"),
)
_STATUSES = {
    "eligible",
    "insufficient_duration",
    "missing_metrics",
    "memory_limit",
    "oom",
    "timeout",
    "failed",
    "interrupted",
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status(trial: Mapping[str, Any]) -> str:
    value = str(trial.get("status", "unknown")).strip().lower()
    return value if value in _STATUSES else (value or "unknown")


def _batch_size(trial: Mapping[str, Any]) -> float | None:
    value = _finite(trial.get("batch_size"))
    return value if value is not None and value > 0 else None


def _recommendation_batch(recommendation: Mapping[str, Any] | None) -> float | None:
    if not recommendation:
        return None
    return _batch_size(recommendation)


def _plot_metric(
    axis: Any,
    trials: Sequence[Mapping[str, Any]],
    field: str,
    title: str,
    color: str,
    target: float | None,
) -> None:
    plotted = False
    for trial in trials:
        batch = _batch_size(trial)
        value = _finite(trial.get(field))
        if batch is None or value is None:
            continue
        status = _status(trial)
        excluded = status != "eligible"
        axis.scatter(
            batch,
            value,
            s=46,
            color=color,
            alpha=0.42 if excluded else 0.95,
            marker="x" if excluded else "o",
            linewidths=1.4,
            label="excluded: " + status if excluded else "eligible",
        )
        plotted = True

    axis.set_title(title, loc="left", fontsize=10, weight="bold", pad=8)
    axis.set_xscale("log", base=2)
    axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cbd5e1")
    axis.tick_params(labelsize=8)
    if target is not None and math.isfinite(target):
        axis.axhline(target, color="#64748b", linestyle=(0, (4, 3)), linewidth=1.0,
                     label=f"target {target:g}")
    if not plotted:
        axis.text(0.5, 0.5, "No finite saved metric", transform=axis.transAxes,
                  ha="center", va="center", color="#64748b", fontsize=9)
    else:
        handles, labels = axis.get_legend_handles_labels()
        # Collapse repeated point labels while retaining a compact status note.
        unique: dict[str, Any] = {}
        for handle, label in zip(handles, labels):
            unique.setdefault(label, handle)
        axis.legend(unique.values(), unique.keys(), fontsize=7, frameon=False, loc="best")


def generate_tuning_plot(
    trials: list[dict],
    recommendation: dict | None,
    output_path: Path,
    *,
    target_utilization: float = 95,
    max_memory_fraction: float = 0.85,
) -> None:
    """Write a polished, headless PNG comparing saved batch-size trials.

    Finite values are plotted exactly as supplied, including values from
    non-eligible trials.  Missing or non-finite fields are omitted; no values
    are interpolated or inferred.  A recommendation is marked only when its
    batch size is positive and finite.
    """
    if not isinstance(trials, list):
        raise TypeError("trials must be a list of trial dictionaries")
    if any(not isinstance(trial, Mapping) for trial in trials):
        raise TypeError("each trial must be a mapping")
    target = _finite(target_utilization)
    memory_target = _finite(max_memory_fraction)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, dpi=160)
    figure.patch.set_facecolor("#f8fafc")
    for axis, (field, title, color) in zip(axes, _METRICS):
        threshold = memory_target if field == "peak_memory_fraction" else target if field == "min_rank_utilization_pct" else None
        _plot_metric(axis, trials, field, title, color, threshold)
    for trial in trials:
        batch, board = _batch_size(trial), _finite(trial.get("board_used_fraction"))
        if batch is not None and board is not None:
            axes[-1].scatter(batch, board, marker="s", facecolors="none", edgecolors="#475569",
                             s=42, label="sampled board use")
    axes[-1].set_xlabel("Batch size (log₂ scale)", fontsize=9)

    recommended = _recommendation_batch(recommendation)
    if recommended is not None:
        for axis in axes:
            axis.axvline(recommended, color="#db2777", linestyle="--", linewidth=1.2,
                         label=f"recommended {recommended:g}")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize=7, frameon=False, loc="best")

    counts = Counter(_status(trial) for trial in trials)
    notes = ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())) or "no trials"
    figure.suptitle("Batch-size autotuning", x=0.08, ha="left", fontsize=14, weight="bold")
    figure.text(0.08, 0.015,
                "Observed sampled utilization; this does not guarantee hardware compute saturation. "
                f"Trial statuses: {notes}", fontsize=8, color="#475569")
    figure.tight_layout(rect=(0, 0.055, 1, 0.96))

    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=".png", dir=str(destination.parent)
        )
        os.close(descriptor)
        figure.savefig(temporary_name, format="png", bbox_inches="tight")
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        plt.close(figure)
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
