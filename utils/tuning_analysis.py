"""Pure, offline analysis for selecting a measured batch-size trial.

This module deliberately depends only on the standard library.  It never imports
the training stack or attempts to infer a successful trial from missing telemetry.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


_MISSING = {"", "n/a", "na", "none", "null", "-", "[not supported]"}
_REQUIRED = {
    "timestamp", "window", "rank", "world_size", "device_type", "device_uuid", "device_name",
    "batch_count", "sample_count", "elapsed_seconds", "window_seconds", "forward_flops",
    "training_flops", "extra_flops", "total_flops", "gpu_util_pct", "gpu_mem_used_gib",
    "gpu_mem_total_gib", "cuda_peak_reserved_gib",
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    try:
        number = Decimal(str(value).strip())
        if not number.is_finite() or number != number.to_integral_value():
            return None
        return int(number)
    except (InvalidOperation, TypeError, ValueError):
        return None


def _timestamp(value: Any) -> dt.datetime | None:
    try:
        result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return result.replace(tzinfo=result.tzinfo or dt.timezone.utc).astimezone(dt.timezone.utc)


def _uuid(value: Any) -> str:
    return str(value or "").strip().lower().removeprefix("gpu-")


def _expected_work(config: Mapping[str, Any], batch_size: int) -> tuple[int, int, int] | None:
    """Return (forward, training, extra) counts when workload metadata is complete."""
    names = ("input_dim", "hidden_dim", "output_dim", "num_hidden_layers")
    values = [_integer(config.get(name)) for name in names]
    if any(value is None or value <= 0 for value in values):
        return None
    input_dim, hidden_dim, output_dim, layers = values
    forward = 2 * input_dim * hidden_dim + (layers - 1) * 2 * hidden_dim * hidden_dim + 2 * hidden_dim * output_dim
    training = 4 * input_dim * hidden_dim + (layers - 1) * 6 * hidden_dim * hidden_dim + 6 * hidden_dim * output_dim
    repeats = _integer(config.get("extra_matmul_repeats", 0))
    extra_dim = _integer(config.get("extra_matmul_dim", hidden_dim))
    if repeats is None or repeats < 0 or extra_dim is None or extra_dim <= 0:
        return None
    return batch_size * forward, batch_size * training, 2 * repeats * extra_dim ** 3


def _present(value: Any) -> bool:
    return value is not None and str(value).strip().lower() not in _MISSING


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Run configuration must be a JSON object")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Metrics CSV must have a nonempty, unique header row")
        missing = _REQUIRED - set(headers)
        if missing:
            raise ValueError(f"Metrics CSV is missing required fields: {', '.join(sorted(missing))}")
        rows = []
        for source in reader:
            if None in source:
                rows.append({"_invalid": "extra CSV fields"})
                continue
            row: dict[str, Any] = dict(source)
            for field in ("window", "rank", "world_size", "batch_count", "sample_count"):
                row[field] = _integer(source.get(field))
            for field in ("elapsed_seconds", "window_seconds", "forward_flops", "training_flops",
                          "extra_flops", "total_flops", "gpu_util_pct",
                          "cuda_peak_reserved_gib", "gpu_mem_total_gib", "gpu_mem_used_gib", "gpu_temp_c"):
                row[field] = _integer(source.get(field)) if field in {
                    "forward_flops", "training_flops", "extra_flops", "total_flops"
                } else _number(source.get(field))
            row["_timestamp"] = _timestamp(source.get("timestamp"))
            row["device_uuid"] = _uuid(source.get("device_uuid"))
            row["_invalid"] = any(
                _present(source.get(field)) and row[field] is None
                for field in ("window", "rank", "world_size", "batch_count", "sample_count",
                              "elapsed_seconds", "window_seconds", "forward_flops", "training_flops",
                              "extra_flops", "total_flops",
                              "gpu_util_pct", "cuda_peak_reserved_gib", "gpu_mem_total_gib",
                              "gpu_mem_used_gib") or row["_timestamp"] is None
            )
            rows.append(row)
    return rows


def analyze_trial(
    metrics_path: Path,
    config_path: Path,
    target_utilization: float = 95,
    max_memory_fraction: float = 0.85,
    min_measured_seconds: float = 8,
) -> dict[str, Any]:
    """Summarize one completed trial without running any workload.

    Invalid or incomplete windows are excluded from the measured aggregate and
    recorded in ``reasons``.  A recommendation therefore cannot be made from a
    partial rank set or fabricated zero telemetry.
    """
    target = _number(target_utilization)
    cap = _number(max_memory_fraction)
    minimum = _number(min_measured_seconds)
    if target is None or not 0 <= target <= 100 or cap is None or not 0 < cap <= 1:
        raise ValueError("target_utilization must be 0..100 and max_memory_fraction must be 0..1")
    if minimum is None or minimum < 0:
        raise ValueError("min_measured_seconds must be finite and non-negative")
    config = _load_config(Path(config_path))
    runtime = config.get("runtime")
    if not isinstance(runtime, dict) or config.get("status") != "completed":
        return {
            "status": "missing_metrics", "batch_size": _integer(config.get("batch_size")),
            "weighted_samples_per_second": None, "weighted_tflops": None,
            "min_rank_utilization_pct": None, "target_met": False,
            "peak_memory_fraction": None, "board_used_fraction": None,
            "board_memory_limit_exceeded": False, "measured_seconds": 0.0, "window_count": 0,
            "per_rank": {}, "reasons": ["configuration status is not completed or runtime metadata is missing"],
        }
    batch_size = _integer(config.get("batch_size"))
    if batch_size is None or batch_size < 1:
        raise ValueError("Run configuration has no valid positive batch_size")
    expected_work = _expected_work(config, batch_size)
    rows = _load_rows(Path(metrics_path))
    expected = _integer(runtime.get("world_size"))
    if expected is None or expected < 1:
        raise ValueError("Run configuration runtime.world_size must be a positive integer")
    gpu_map: dict[int, dict[str, Any]] = {}
    raw_gpus = runtime.get("gpus", [])
    if isinstance(raw_gpus, list):
        for item in raw_gpus:
            if isinstance(item, dict) and _integer(item.get("logical_index")) is not None:
                gpu_map[_integer(item["logical_index"])] = item
    expected_uuids = {_uuid(item.get("uuid")): rank for rank, item in gpu_map.items() if _uuid(item.get("uuid"))}
    if (not gpu_map or len(expected_uuids) != len(gpu_map) or set(gpu_map) != set(range(expected))):
        reasons = ["runtime GPU logical-index mapping is missing, incomplete, or has duplicate UUIDs"]
    else:
        reasons = []
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("_invalid"):
            reasons.append("invalid CSV row")
            continue
        if row["window"] is None:
            reasons.append("invalid window identifier")
            continue
        grouped[row["window"]].append(row)
    valid: list[tuple[int, list[dict[str, Any]], float]] = []
    for window, members in sorted(grouped.items()):
        local: list[str] = []
        world_sizes = {row["world_size"] for row in members}
        worlds = [value for value in world_sizes if value is not None and value > 0]
        if world_sizes != {expected} or len(worlds) != 1:
            local.append(f"window {window} has inconsistent world size")
            reasons.extend(local)
            continue
        world = worlds[0]
        if world != expected:
            local.append(f"window {window} disagrees with world size")
        ranks = [row["rank"] for row in members]
        if len(members) != world or set(ranks) != set(range(world)):
            local.append(f"window {window} is missing or duplicating ranks")
        if len(set(ranks)) != len(ranks):
            local.append(f"window {window} has duplicate ranks")
        if any(str(row["device_type"] or "").strip().lower() != "cuda" for row in members):
            local.append(f"window {window} contains non-CUDA telemetry")
        if gpu_map:
            for row in members:
                saved = gpu_map.get(row["rank"], {})
                if _uuid(saved.get("uuid")) != row["device_uuid"]:
                    local.append(f"window {window} has unstable GPU UUID mapping")
        if len({row["device_uuid"] for row in members}) != expected:
            local.append(f"window {window} has duplicate or missing GPU UUIDs")
        for row in members:
            valid_counts = (row["batch_count"] is not None and row["batch_count"] >= 1
                            and row["sample_count"] == row["batch_count"] * batch_size)
            if not valid_counts:
                local.append(f"window {window} has invalid batch or sample counts")
                continue
            duration = row["elapsed_seconds"]
            if (not valid_counts or duration is None or duration <= 0 or row["window_seconds"] is None
                    or row["window_seconds"] <= 0 or row["total_flops"] is None or row["total_flops"] <= 0
                    or row["training_flops"] is None or row["extra_flops"] is None
                    or row["forward_flops"] is None or row["total_flops"] != row["training_flops"] + row["extra_flops"]
                    or row["forward_flops"] > row["training_flops"]):
                local.append(f"window {window} has invalid counts, duration, or FLOPs")
            expected_forward, expected_training, expected_extra = expected_work or (None, None, None)
            expected_row_work = (
                row["batch_count"] * expected_forward if expected_forward is not None else None,
                row["batch_count"] * expected_training if expected_training is not None else None,
                row["batch_count"] * expected_extra if expected_extra is not None else None,
            )
            if expected_work is None or (row["forward_flops"], row["training_flops"], row["extra_flops"]) != expected_row_work:
                local.append(f"window {window} FLOPs disagree with saved workload configuration")
            if (row["gpu_mem_used_gib"] is None or row["gpu_mem_total_gib"] is None
                    or row["gpu_mem_total_gib"] <= 0 or row["gpu_mem_used_gib"] < 0
                    or row["gpu_mem_used_gib"] > row["gpu_mem_total_gib"]):
                local.append(f"window {window} has incomplete board memory telemetry")
            if (row["cuda_peak_reserved_gib"] is None or row["cuda_peak_reserved_gib"] <= 0
                    or row["gpu_mem_total_gib"] is None
                    or row["cuda_peak_reserved_gib"] > row["gpu_mem_total_gib"]):
                local.append(f"window {window} has invalid reserved memory telemetry")
        if local:
            reasons.extend(local)
            continue
        duration = max(row["elapsed_seconds"] for row in members)
        if any(not math.isclose(row["window_seconds"], duration, rel_tol=1e-8, abs_tol=1e-9) for row in members):
            reasons.append(f"window {window} has inconsistent rank durations")
            continue
        if len({row["batch_count"] for row in members}) != 1:
            reasons.append(f"window {window} has unaligned batch counts")
            continue
        valid.append((window, members, duration))

    measured_seconds = sum(duration for _, _, duration in valid)
    samples = sum(sum(row["sample_count"] for row in members) for _, members, _ in valid)
    flops = sum(sum(row["total_flops"] for row in members) for _, members, _ in valid)
    weighted_samples = samples / measured_seconds if measured_seconds > 0 else None
    weighted_tflops = flops / measured_seconds / 1e12 if measured_seconds > 0 else None
    per_rank: dict[str, dict[str, Any]] = {}
    memory_fractions: list[float] = []
    board_fractions: list[float] = []
    for rank in range(expected or 0):
        util_area = util_seconds = 0.0
        unknown_windows = 0
        memory_ok = True
        rank_memory: list[float] = []
        temperatures: list[float] = []
        names: set[str] = set()
        rank_rows = [(next(row for row in members if row["rank"] == rank), duration)
                     for _, members, duration in valid]
        for row, duration in rank_rows:
            util = row["gpu_util_pct"]
            if util is None or not 0 <= util <= 100:
                unknown_windows += 1
            else:
                util_area += util * duration
                util_seconds += duration
            reserved, total, used = row["cuda_peak_reserved_gib"], row["gpu_mem_total_gib"], row["gpu_mem_used_gib"]
            if reserved is None or total is None or total <= 0 or reserved < 0 or reserved > total:
                memory_ok = False
            else:
                memory_fractions.append(reserved / total)
                rank_memory.append(reserved / total)
            if used is not None and total is not None and total > 0 and 0 <= used <= total:
                board_fractions.append(used / total)
            if row.get("gpu_temp_c") is not None:
                temperatures.append(row["gpu_temp_c"])
            names.add(str(row.get("device_name") or ""))
        utilization = util_area / util_seconds if util_seconds else None
        per_rank[str(rank)] = {
            "utilization_pct": utilization,
            "telemetry_available": utilization is not None and unknown_windows == 0,
            "unknown_telemetry_windows": unknown_windows,
            "memory_telemetry_complete": memory_ok and bool(valid),
            "peak_memory_fraction": max(rank_memory, default=None),
            "max_temperature_c": max(temperatures, default=None),
            "device_uuid": _uuid(rank_rows[0][0].get("device_uuid")) if rank_rows else None,
            "device_name": sorted(names)[0] if names else None,
            "measurement_count": len(rank_rows),
        }
    min_util = min((item["utilization_pct"] for item in per_rank.values() if item["utilization_pct"] is not None), default=None)
    target_met = bool(per_rank) and all(
        item["telemetry_available"] and item["utilization_pct"] >= target
        for item in per_rank.values()
    )
    peak_memory = max(memory_fractions) if memory_fractions else None
    board_pressure = max(board_fractions, default=None)
    memory_complete = bool(valid) and all(item["memory_telemetry_complete"] for item in per_rank.values())
    util_complete = bool(valid) and len(valid) >= 3 and all(
        sum(1 for _, members, _ in valid
            if next(row for row in members if row["rank"] == rank)["gpu_util_pct"] is not None
            and 0 <= next(row for row in members if row["rank"] == rank)["gpu_util_pct"] <= 100) >= 3
        and all(next(row for row in members if row["rank"] == rank)["gpu_util_pct"] is not None
                and 0 <= next(row for row in members if row["rank"] == rank)["gpu_util_pct"] <= 100
                for _, members, _ in valid)
        for rank in range(expected)
    )
    invalid_evidence = bool(reasons)
    if not valid or invalid_evidence or not util_complete:
        status = "missing_metrics"
        if not util_complete:
            reasons.append("at least three complete windows with utilization telemetry per rank are required")
    elif not memory_complete or peak_memory is None:
        status = "missing_metrics"
        reasons.append("complete CUDA memory telemetry is unavailable")
    elif peak_memory > cap + 1e-12:
        status = "memory_limit"
        reasons.append(f"peak reserved memory fraction exceeds {cap:.3f}")
    elif measured_seconds + 1e-9 < minimum:
        status = "insufficient_duration"
        reasons.append(f"measured duration is below {minimum:g} seconds")
    else:
        status = "eligible"
    if board_pressure is not None and board_pressure > cap + 1e-12:
        if status != "missing_metrics":
            status = "memory_limit"
        reasons.append("whole-board memory usage exceeds the memory limit (includes CUDA/NCCL and other nonallocator usage)")
    return {
        "status": status, "batch_size": batch_size, "weighted_samples_per_second": weighted_samples,
        "weighted_tflops": weighted_tflops, "min_rank_utilization_pct": min_util,
        "target_met": target_met, "peak_memory_fraction": peak_memory,
        "board_used_fraction": board_pressure, "board_memory_limit_exceeded": board_pressure is not None and board_pressure > cap + 1e-12,
        "measured_seconds": measured_seconds, "window_count": len(valid), "per_rank": per_rank,
        "reasons": sorted(set(reasons)),
    }


def select_recommendation(trials: Sequence[Mapping[str, Any]], throughput_tolerance: float = .03) -> dict[str, Any] | None:
    """Choose the smallest batch within tolerance of the global fastest trial.

    Utilization target status is a tie-breaker among near-fastest candidates; it
    cannot make a materially slower trial win.
    """
    tolerance = _number(throughput_tolerance)
    if tolerance is None or not 0 <= tolerance < 1:
        raise ValueError("throughput_tolerance must be finite and in [0, 1)")
    eligible = [dict(trial) for trial in trials if trial.get("status") == "eligible"
                and (_number(trial.get("weighted_samples_per_second")) or 0) > 0
                and (_integer(trial.get("batch_size")) or 0) > 0]
    if not eligible:
        return None
    fastest = max(float(trial["weighted_samples_per_second"]) for trial in eligible)
    near = [trial for trial in eligible if float(trial["weighted_samples_per_second"]) >= fastest * (1 - tolerance)]
    target = [trial for trial in near if trial.get("target_met") is True]
    candidates = target or near
    selected = min(candidates, key=lambda trial: (int(trial["batch_size"]), -float(trial["weighted_samples_per_second"])))
    result = dict(selected)
    result["selection_reason"] = (
        "smallest target-met batch within global fastest throughput tolerance"
        if target else "smallest eligible batch within global fastest throughput tolerance"
    )
    return result
