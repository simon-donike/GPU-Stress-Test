"""Offline reports from saved CSV/JSON artifacts; never imports the training stack."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import tempfile
import textwrap
import uuid
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DPI = 160
COLORS = ("#2563eb", "#d97706", "#059669", "#db2777", "#7c3aed", "#0891b2", "#dc2626", "#64748b")
COUNT_FIELDS = {
    "window", "rank", "world_size", "batch_count", "sample_count", "forward_flops",
    "training_flops", "extra_flops", "total_flops",
}
VALUE_FIELDS = {
    "elapsed_seconds", "window_seconds", "tflops_per_device", "tflops_total",
    "samples_per_second_total", "gpu_temp_c", "gpu_util_pct", "gpu_mem_used_gib",
    "gpu_mem_total_gib", "gpu_mem_used_pct", "gpu_power_w", "gpu_power_limit_w",
    "cuda_peak_allocated_gib", "cuda_peak_reserved_gib",
}
REQUIRED_FIELDS = COUNT_FIELDS | {"timestamp", "elapsed_seconds", "window_seconds"}
MISSING_VALUES = {"", "n/a", "na", "none", "null", "-", "[not supported]"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    try:
        number = Decimal(str(value))
        if number.is_finite() and number == number.to_integral_value():
            return int(number)
    except (InvalidOperation, ValueError):
        pass
    return None


def _timestamp(value: Any) -> dt.datetime | None:
    try:
        timestamp = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(dt.timezone.utc)


def _load_metrics(path: Path) -> tuple[list[dict[str, Any]], bool]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if not headers or len(headers) != len(set(headers)):
            raise ValueError("Metrics CSV must have a nonempty, unique header row.")
        modern = bool({"window", "rank", "total_flops"} & set(headers))
        required = REQUIRED_FIELDS if modern else {"timestamp", "step", "tflops_total"}
        missing = required - set(headers)
        if missing:
            raise ValueError(f"Metrics CSV is missing required fields: {', '.join(sorted(missing))}")
        rows = []
        for index, source in enumerate(reader):
            row: dict[str, Any] = dict(source)
            invalid_numeric = 0
            for field in COUNT_FIELDS | VALUE_FIELDS:
                value = source.get(field)
                parsed = _integer(value) if field in COUNT_FIELDS else _number(value)
                row[field] = parsed
                if value is not None and str(value).strip().lower() not in MISSING_VALUES and parsed is None:
                    invalid_numeric += 1
            if not modern:
                row["window"] = source.get("step", str(index))
                row["rank"] = source.get("gpu_id", "unknown")
                row["device_name"] = source.get("gpu_name", "unknown")
                row["gpu_mem_used_gib"] = _number(source.get("gpu_mem_used_gb"))
                row["tflops_per_device"] = _number(source.get("tflops_per_gpu"))
            row["_time"] = _timestamp(source.get("timestamp"))
            row["_index"] = index
            row["_invalid_numeric"] = invalid_numeric
            row["_malformed"] = None in source or any(value is None for value in source.values())
            rows.append(row)
    rows.sort(key=lambda row: (row["_time"] is None,
                              row["_time"] or dt.datetime.max.replace(tzinfo=dt.timezone.utc), row["_index"]))
    return rows, not modern


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Run configuration must be a JSON object.")
    return value


def _aggregate(rows: Sequence[Mapping[str, Any]], legacy: bool, expected_world_size: int | None = None) -> dict[str, Any]:
    grouped: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["window"]].append(row)
    points = []
    issues = []
    duplicate_rows = 0
    sizes = Counter(row["world_size"] for row in rows if row["world_size"] is not None and row["world_size"] > 0)
    common_sizes = sizes.most_common(2)
    expected = expected_world_size
    if expected is None and common_sizes and (len(common_sizes) == 1 or common_sizes[0][1] > common_sizes[1][1]):
        expected = common_sizes[0][0]
    for window, members in grouped.items():
        representative = members[0]
        if legacy:
            # Historical rows provide no measured counts/durations. Plot stored rates
            # once per step, but never manufacture a time-weighted run aggregate.
            rates = [row["tflops_total"] for row in members]
            if (any(row["_malformed"] or row["_time"] is None for row in members)
                    or any(rate is None or rate < 0 for rate in rates)):
                issues.append({"window": window, "reasons": ["invalid legacy timestamp, rate, or CSV row"]})
                continue
            if any(not math.isclose(rate, rates[0], rel_tol=1e-6, abs_tol=1e-9) for rate in rates):
                issues.append({"window": window, "reasons": ["inconsistent repeated legacy global rate"]})
                continue
            points.append({"window": window, "time": representative["_time"],
                           "tflops": rates[0] if rates else None})
            continue
        reasons = []
        ranks = [row["rank"] for row in members]
        duplicates = len(ranks) - len(set(ranks))
        duplicate_rows += duplicates
        if expected is None or expected < 1:
            reasons.append("inconsistent or invalid world size")
        elif len(ranks) != expected or any(rank is None or rank < 0 or rank >= expected for rank in ranks):
            reasons.append("missing or invalid rank")
        if any(row["world_size"] != expected for row in members):
            reasons.append("world size disagrees with saved configuration or other windows")
        if duplicates:
            reasons.append("duplicate rank")
        if window is None or window < 0:
            reasons.append("invalid window identifier")
        for row in members:
            counts_valid = all(isinstance(row[field], int) and row[field] >= 0 for field in COUNT_FIELDS)
            seconds = row["elapsed_seconds"]
            if (row["_malformed"] or not counts_valid or row["_time"] is None
                    or seconds is None or seconds <= 0
                    or row["window_seconds"] is None or row["window_seconds"] <= 0):
                reasons.append("invalid counts, duration, timestamp, or CSV row")
                continue
            if (row["batch_count"] < 1 or row["sample_count"] < 1 or row["training_flops"] <= 0
                    or row["forward_flops"] > row["training_flops"]
                    or row["total_flops"] != row["training_flops"] + row["extra_flops"]):
                reasons.append("inconsistent work counts")
        if not reasons:
            duration = max(row["elapsed_seconds"] for row in members)
            if any(not math.isclose(row["window_seconds"], duration, rel_tol=1e-8, abs_tol=1e-9) for row in members):
                reasons.append("global duration disagrees with rank durations")
            if len({row["batch_count"] for row in members}) != 1:
                reasons.append("unaligned rank batch counts")
        if reasons:
            issues.append({"window": window, "reasons": sorted(set(reasons))})
            continue
        total = sum(row["total_flops"] for row in members)
        samples = sum(row["sample_count"] for row in members)
        points.append({"window": window, "time": representative["_time"], "total_flops": total,
                       "seconds": duration, "samples": samples, "tflops": total / duration / 1e12})
    if not legacy:
        points.sort(key=lambda point: point["window"])
    total_flops = sum(point.get("total_flops", 0) for point in points)
    total_seconds = sum(point.get("seconds", 0.0) for point in points)
    samples = sum(point.get("samples", 0) for point in points)
    return {
        "points": points, "window_count": len(points), "input_window_count": len(grouped),
        "expected_ranks": expected, "excluded_windows": issues, "duplicate_rows": duplicate_rows,
        "total_flops": total_flops if not legacy else None,
        "total_window_seconds": total_seconds if not legacy else None,
        "aggregate_tflops": total_flops / total_seconds / 1e12 if total_seconds else None,
        "sample_count": samples if not legacy else None,
        "samples_per_second": samples / total_seconds if total_seconds else None,
    }


def _rank_order(rank: Any) -> tuple[int, Any]:
    number = _integer(rank)
    return (0, number) if number is not None else (1, str(rank))


def _decorate_axis(ax: Any, title: str, unit: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, weight="bold", pad=10)
    ax.set_ylabel(unit, fontsize=9)
    ax.grid(axis="y", color="#e2e8f0", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=dt.timezone.utc))
    ax.tick_params(labelsize=8)


def _plot_metric(ax: Any, rows: Sequence[Mapping[str, Any]], fields: Sequence[tuple[str, str]],
                 title: str, unit: str, colors: Mapping[Any, str]) -> None:
    _decorate_axis(ax, title, unit)
    any_data = False
    for rank in sorted(colors, key=_rank_order):
        for field_index, (field, label) in enumerate(fields):
            points = [(row["_time"], row[field]) for row in rows
                      if row["rank"] == rank and row["_time"] is not None and row.get(field) is not None]
            if not points:
                continue
            any_data = True
            xs, ys = zip(*points)
            name = f"rank {rank}" + (f" {label}" if len(fields) > 1 else "")
            ax.plot(xs, ys, color=colors[rank], linestyle="--" if field_index else "-",
                    marker="o", markersize=3, linewidth=1.4, label=name)
    if any_data:
        ax.legend(fontsize=7, ncol=2, frameon=False)
        ax.set_ylim(bottom=0)
    else:
        ax.text(.5, .5, "No saved readings", transform=ax.transAxes, ha="center", color="#64748b")
        ax.set_xticks([])
        ax.set_yticks([])


def _dashboard(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
               aggregate: Mapping[str, Any], legacy: bool) -> Any:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12), dpi=DPI)
    fig.patch.set_facecolor("#f8fafc")
    runtime = config.get("runtime", {})
    precision = runtime.get("precision", config.get("precision", "unknown")) if isinstance(runtime, dict) else "unknown"
    title = "GPU STRESS TEST  /  saved run"
    subtitle = f"{config.get('run_name', 'unnamed')}  |  {config.get('status', 'unknown status')}  |  precision: {precision}"
    fig.text(.07, .974, title, fontsize=18, weight="bold", color="#0f172a")
    fig.text(.07, .950, subtitle[:145], fontsize=10, color="#475569")
    colors = {rank: COLORS[index % len(COLORS)] for index, rank in enumerate(sorted({row["rank"] for row in rows}, key=_rank_order))}
    ax = axes[0, 0]
    _decorate_axis(ax, "Global estimated matmul throughput", "TFLOP/s")
    points = [point for point in aggregate["points"] if point["time"] is not None and point["tflops"] is not None]
    if points:
        ax.plot([point["time"] for point in points], [point["tflops"] for point in points],
                color=COLORS[0], marker="o", markersize=3, linewidth=1.6)
        ax.set_ylim(bottom=0)
    else:
        ax.text(.5, .5, "No valid measured windows", transform=ax.transAxes, ha="center", color="#64748b")
        ax.set_xticks([])
        ax.set_yticks([])
    # Recompute local rates from valid saved counts rather than trusting derived columns.
    plot_rows = [dict(row) for row in rows]
    if not legacy:
        valid_windows = {point["window"] for point in aggregate["points"]}
        for row in plot_rows:
            row["tflops_per_device"] = (row["total_flops"] / row["elapsed_seconds"] / 1e12
                                        if row["window"] in valid_windows else None)
    specs = [
        (axes[0, 1], [("tflops_per_device", "")], "Per-device estimated throughput", "TFLOP/s"),
        (axes[1, 0], [("gpu_util_pct", "")], "Sampled GPU utilization", "%"),
        (axes[1, 1], [("gpu_temp_c", "")], "Sampled GPU temperature", "C"),
        (axes[2, 0], [("gpu_mem_used_gib", "")], "Board memory used", "GiB"),
        (axes[2, 1], [("gpu_power_w", "")], "Sampled GPU power", "W"),
        (axes[3, 0], [("cuda_peak_allocated_gib", "allocated"), ("cuda_peak_reserved_gib", "reserved")],
         "Per-window allocator peaks", "GiB"),
    ]
    for axis, fields, label, unit in specs:
        _plot_metric(axis, plot_rows, fields, label, unit, colors)
    axes[3, 1].axis("off")
    rate = aggregate["aggregate_tflops"]
    count_label = "Valid legacy steps" if legacy else "Complete windows"
    lines = ["MEASUREMENT SUMMARY", "", f"{count_label}: {aggregate['window_count']}",
             f"Excluded windows: {len(aggregate['excluded_windows'])}",
             f"Weighted rate: {rate:.3f} TFLOP/s" if rate is not None else "Weighted rate: unavailable",
             f"Samples: {aggregate['sample_count']:,}" if aggregate["sample_count"] is not None else "Samples: unknown",
             "FMA = 2; counted dense matmuls only.", "Telemetry is sampled, not a continuous maximum."]
    if legacy:
        lines.append("LEGACY / UNVERIFIED FLOP ESTIMATES")
    axes[3, 1].text(.03, .95, "\n".join(lines), va="top", fontsize=9, linespacing=1.3, color="#334155")
    fig.text(.07, .018, "UTC save times  |  Warmup excluded  |  Training and host time included; telemetry and report work excluded", fontsize=8, color="#64748b")
    fig.subplots_adjust(left=.075, right=.97, top=.91, bottom=.065, hspace=.58, wspace=.23)
    return fig


def _config_rows(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, dict):
        rows = []
        for key, item in value.items():
            rows.extend(_config_rows(item, f"{prefix}.{key}" if prefix else str(key)))
        return rows
    if isinstance(value, list):
        rows = []
        for index, item in enumerate(value):
            rows.extend(_config_rows(item, f"{prefix}[{index}]"))
        return rows or [(prefix, "[]")]
    return [(prefix, str(value))]


def _metadata_pages(pdf: PdfPages, config: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
    # Draw wrapped text with explicit row heights: long nested metadata must not
    # overflow a table or disappear behind an ellipsis.
    lines = []
    for key, value in _config_rows({"run": config, "summary": summary}):
        keys = textwrap.wrap(key, 38) or [""]
        values = textwrap.wrap(value, 66) or [""]
        height = max(len(keys), len(values))
        lines.append((keys, values, height))
    pages: list[list[Any]] = [[]]
    used = 0
    for keys, values, height in lines:
        for offset in range(0, height, 45):
            segment = (keys[offset:offset + 45], values[offset:offset + 45], min(45, height - offset))
            if used + segment[2] + 1 > 48:
                pages.append([])
                used = 0
            pages[-1].append(segment)
            used += segment[2] + 1
    for page_number, page in enumerate(pages, 1):
        fig = plt.figure(figsize=(8.27, 11.69))
        try:
            fig.text(.07, .95, f"Saved run metadata  /  {page_number}", fontsize=15, weight="bold", color="#0f172a")
            y = .90
            for keys, values, height in page:
                fig.text(.07, y, "\n".join(keys), fontsize=7, va="top", color="#475569", linespacing=1.25)
                fig.text(.40, y, "\n".join(values), fontsize=7, va="top", color="#0f172a", linespacing=1.25)
                y -= (height + 1) * .016
            pdf.savefig(fig)
        finally:
            plt.close(fig)


def generate_report(metrics_path: Path, config_path: Path, output_pdf: Path) -> None:
    """Write a PDF, summary.png, and summary.json from saved artifacts only."""
    metrics_path, config_path, output_pdf = map(Path, (metrics_path, config_path, output_pdf))
    if output_pdf.suffix.lower() != ".pdf":
        raise ValueError("Report output must use a .pdf extension.")
    dashboard_path = output_pdf.parent / "summary.png"
    summary_path = output_pdf.parent / "summary.json"
    inputs = {metrics_path.resolve(), config_path.resolve()}
    if any(path.resolve() in inputs for path in (output_pdf, dashboard_path, summary_path)):
        raise ValueError("Report outputs must not overwrite the input metrics or configuration.")
    rows, legacy = _load_metrics(metrics_path)
    config = _load_config(config_path)
    runtime = config.get("runtime", {})
    expected = _integer(runtime.get("world_size")) if isinstance(runtime, dict) else None
    aggregate = _aggregate(rows, legacy, expected if expected is not None and expected > 0 else None)
    maxima = {}
    for field in ("gpu_temp_c", "gpu_util_pct", "gpu_mem_used_gib", "gpu_mem_used_pct", "gpu_power_w",
                  "cuda_peak_allocated_gib", "cuda_peak_reserved_gib"):
        values = [row[field] for row in rows if row[field] is not None]
        maxima[field] = max(values) if values else None
    summary = {
        "schema_version": 2, "metrics_file": metrics_path.name, "config_file": config_path.name,
        "report_generation_id": uuid.uuid4().hex,
        "run_status": config.get("status", "unknown"),
        "schema": "legacy/unverified" if legacy else "windowed/per-rank",
        "flop_accounting": "historical unverified estimate" if legacy else "analytical dense matmul estimate; FMA = 2",
        **{key: value for key, value in aggregate.items() if key != "points"},
        "invalid_numeric_fields": sum(row["_invalid_numeric"] for row in rows),
        "max_saved_readings": maxima,
    }
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    # Stage all artifacts before replacing older report outputs. Input data stays untouched.
    with tempfile.TemporaryDirectory(prefix=".report-", dir=output_pdf.parent) as staging:
        pdf_temp = Path(staging) / "report.pdf"
        png_temp = Path(staging) / "summary.png"
        json_temp = Path(staging) / "summary.json"
        with plt.rc_context({"font.family": "DejaVu Sans", "axes.facecolor": "white"}):
            fig = _dashboard(rows, config, aggregate, legacy)
            try:
                fig.savefig(png_temp, format="png", dpi=DPI)
                with PdfPages(pdf_temp) as pdf:
                    pdf.infodict().update(Title="GPU Stress Test: saved run", Subject="Analytical matmul throughput and sampled GPU telemetry")
                    pdf.savefig(fig)
                    # These fields describe the surrounding generation attempt,
                    # which has not completed when the source JSON is read.
                    report_config = {key: value for key, value in config.items()
                                     if key not in {"report_status", "report_error"}}
                    _metadata_pages(pdf, report_config, summary)
            finally:
                plt.close(fig)
        summary["artifact_sha256"] = {
            output_pdf.name: hashlib.sha256(pdf_temp.read_bytes()).hexdigest(),
            dashboard_path.name: hashlib.sha256(png_temp.read_bytes()).hexdigest(),
        }
        json_temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pdf_temp.replace(output_pdf)
        png_temp.replace(dashboard_path)
        json_temp.replace(summary_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an offline PDF/PNG/JSON report from saved run artifacts.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        generate_report(args.metrics, args.config, args.output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"Report failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
