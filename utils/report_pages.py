"""Standardized report pages drawn only from saved, parsed measurement data."""
from __future__ import annotations

import datetime as dt
import math
import textwrap
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from utils import report_style as style


def rank_order(rank: Any) -> tuple[int, Any]:
    try:
        return (0, int(rank))
    except (TypeError, ValueError):
        return (1, str(rank))


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:,.{digits}f}"


def _compact(value: Any) -> str:
    if value is None:
        return "Unavailable"
    for scale, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= scale:
            return f"{value / scale:,.2f}{suffix}"
    return f"{value:,.1f}"


def _ranks(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
    return sorted({row["rank"] for row in rows}, key=rank_order)


def _identity(config: Mapping[str, Any]) -> tuple[str, str]:
    return str(config.get("run_name", "Unnamed saved run")), str(config.get("status", "unknown"))


def _footer(legacy: bool) -> str:
    return ("LEGACY / UNVERIFIED · historical stored rates · no reconstructed timing" if legacy else
            "Estimated dense GEMMs · FMA = 2 · warmup excluded · UTC save times")


def _temperature_limit(config: Mapping[str, Any]) -> float | None:
    limit = config.get("max_temperature_c")
    if isinstance(limit, (int, float)) and not isinstance(limit, bool) and math.isfinite(limit) and 0 < limit <= 150:
        return float(limit)
    return None


def temperature_summary(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Use all parsed, saved temperatures, including excluded throughput windows."""
    values = [row["gpu_temp_c"] for row in rows if row.get("gpu_temp_c") is not None]
    peak = max(values) if values else None
    limit = _temperature_limit(config)
    return {
        "scope": "all_saved_rows", "peak_temperature_c": peak, "configured_stop_c": limit,
        "margin_to_stop_c": limit - peak if limit is not None and peak is not None else None,
        "readable_rows": len(values), "saved_rows": len(rows),
    }


def device_summaries(rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any],
                     legacy: bool) -> list[dict[str, Any]]:
    """Weighted comparisons use only accepted modern windows, never stored rates."""
    valid = {point["window"] for point in aggregate["points"]}
    devices = []
    for rank in _ranks(rows):
        saved = [row for row in rows if row["rank"] == rank]
        measured = [row for row in saved if row["window"] in valid]
        seconds = sum(row["elapsed_seconds"] for row in measured) if not legacy else 0
        total = sum(row["total_flops"] for row in measured) if not legacy else 0
        util = [(row["gpu_util_pct"], row["window_seconds"]) for row in measured
                if not legacy and row.get("gpu_util_pct") is not None and 0 <= row["gpu_util_pct"] <= 100]
        denominator = sum(weight for _, weight in util)
        device = {
            "rank": rank, "device_name": str(saved[0].get("device_name") or "Unknown device"),
            "device_uuid": str(saved[0].get("device_uuid") or "Unavailable"),
            "accepted_windows": len(measured), "weighted_tflops": total / seconds / 1e12 if seconds else None,
            "weighted_utilization_pct": sum(value * weight for value, weight in util) / denominator if denominator else None,
            "utilization_readings": len(util), "utilization_window_count": len(measured) if not legacy else 0,
        }
        for field in ("gpu_temp_c", "gpu_mem_used_gib", "gpu_power_w", "cuda_peak_allocated_gib", "cuda_peak_reserved_gib"):
            values = [row[field] for row in measured if row.get(field) is not None]
            device[f"max_{field}"] = max(values) if values else None
        devices.append(device)
    return devices


def _time_axis(ax: Any, title: str, unit: str) -> None:
    style.axis(ax, title, unit)
    locator = mdates.AutoDateLocator(minticks=2, maxticks=4)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=dt.timezone.utc))
    ax.xaxis.get_offset_text().set_fontsize(7)


def _plot_global(ax: Any, aggregate: Mapping[str, Any], *, samples: bool, legacy: bool) -> None:
    title = "Sample throughput" if samples else "Estimated matmul throughput"
    _time_axis(ax, title, "samples / s" if samples else "TFLOP/s")
    points = [point for point in aggregate["points"] if point["time"] is not None]
    if samples:
        points = [point for point in points if point.get("seconds")]
    if not points:
        style.empty(ax, "Timing unavailable in legacy data" if legacy else "No complete measurement windows")
        return
    # An excluded window interrupts the line: never visually bridge missing work.
    xs, ys = [], []
    previous = None
    for point in points:
        if not legacy and previous is not None and point["window"] != previous + 1:
            xs.append(point["time"])
            ys.append(math.nan)
        xs.append(point["time"])
        ys.append(point["samples"] / point["seconds"] if samples else point["tflops"])
        previous = point["window"]
    color = style.COLORS[2] if samples else style.COLORS[0]
    ax.plot(xs, ys, color=color, marker="o", markersize=3, linewidth=1.7)
    ax.fill_between(xs, ys, alpha=.075, color=color)
    weighted = aggregate["samples_per_second" if samples else "aggregate_tflops"]
    if weighted is not None:
        ax.axhline(weighted, color=style.INK, linestyle=(0, (3, 3)), linewidth=.9,
                   label=f"Weighted run rate  {_compact(weighted) if samples else _fmt(weighted, 2)}")
        ax.legend(loc="best", frameon=False, fontsize=7)
    elif legacy:
        ax.text(.02, .93, "Stored legacy estimate", transform=ax.transAxes, fontsize=8, color=style.MUTED, va="top")
    ax.set_ylim(bottom=0)


def overview(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], aggregate: Mapping[str, Any],
             legacy: bool, thermal: Mapping[str, Any], colors: Mapping[Any, str]) -> Any:
    name, status = _identity(config)
    fig = style.page("Performance & temperature", name, status, 1, note=_footer(legacy))
    peak, limit, margin = thermal["peak_temperature_c"], thermal["configured_stop_c"], thermal["margin_to_stop_c"]
    thermal_detail = "No configured stop · all saved rows"
    if limit is not None:
        thermal_detail = f"Stop {limit:g} °C"
        if margin is not None:
            thermal_detail += (f" · {margin:g} °C below" if margin > 0 else
                               " · reached" if margin == 0 else f" · {-margin:g} °C above")
        else:
            thermal_detail += " · no saved readings"
    thermal_color = style.THERMAL_STOP if margin is not None and margin <= 0 else style.COLORS[1]
    cards = (
        ("Weighted throughput", _fmt(aggregate["aggregate_tflops"], 2), "TFLOP/s · estimated dense matmuls", style.COLORS[0]),
        ("Peak sampled temperature", f"{peak:.1f} °C" if peak is not None else "Unavailable", thermal_detail, thermal_color),
        ("Sample throughput", _compact(aggregate["samples_per_second"]), "samples / second · all ranks", style.COLORS[2]),
        ("Measured time", _fmt(aggregate["total_window_seconds"]), "seconds · sum of slowest-rank windows", style.COLORS[3]),
    )
    for index, (label, value, detail, color) in enumerate(cards):
        style.card(fig, .06 + .227 * index, .65, .208, label, value, detail, color=color)
    throughput_axis = fig.add_axes((.08, .375, .385, .205))
    samples_axis = fig.add_axes((.08, .12, .385, .165))
    thermal_axis = fig.add_axes((.56, .375, .385, .205))
    _plot_global(throughput_axis, aggregate, samples=False, legacy=legacy)
    _plot_global(samples_axis, aggregate, samples=True, legacy=legacy)
    rank_peaks: dict[Any, float] = {}
    for row in rows:
        if row.get("gpu_temp_c") is not None:
            rank_peaks[row["rank"]] = max(rank_peaks.get(row["rank"], -math.inf), row["gpu_temp_c"])
    # Keep the hottest device visible even when a run needs several detail pages.
    hottest = sorted(rank_peaks, key=lambda rank: (-rank_peaks[rank], rank_order(rank)))[:6]
    _plot_temperature(thermal_axis, rows, sorted(hottest, key=rank_order), colors, limit, rank_legend=True)
    times = [row["_time"] for row in rows if row["_time"] is not None]
    if times:
        left, right = min(times), max(times)
        padding = max((right - left) * .025, dt.timedelta(seconds=.5))
        for axis in (throughput_axis, samples_axis, thermal_axis):
            axis.set_xlim(left - padding, right + padding)
    runtime = config.get("runtime", {})
    precision = runtime.get("precision", config.get("precision", "unknown")) if isinstance(runtime, dict) else "unknown"
    note = f"Precision: {precision}. Batch: {config.get('batch_size', 'unknown')} per device. "
    if legacy:
        note += "Legacy rates are unverified; sample and duration totals are unavailable. "
    else:
        note += (f"{aggregate['sample_count']:,} samples; {aggregate['window_count']:,}/{aggregate['input_window_count']:,} "
                 f"accepted windows; {len(aggregate['excluded_windows']):,} excluded. ")
    note += (f"Temperature: {thermal['readable_rows']:,}/{thermal['saved_rows']:,} saved rows readable, "
             "including excluded throughput windows. Sampled peaks are not continuous maxima. ")
    if len(rank_peaks) > len(hottest):
        note += "Hottest six ranks shown; all ranks follow in telemetry pages. "
    if status != "completed":
        note += f"Run status: {status}; saved partial results."
    style.note(fig, .56, .28, "Reading this run", note, width=60)
    return fig


def _series(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[list[Any], list[float]]:
    xs, ys = [], []
    previous = None
    for row in rows:
        window = row.get("window")
        if isinstance(window, int) and isinstance(previous, int) and window != previous + 1:
            xs.append(row["_time"])
            ys.append(math.nan)
        xs.append(row["_time"])
        ys.append(row.get(field) if row.get(field) is not None else math.nan)
        previous = window
    return xs, ys


def _plot_metric(ax: Any, rows: Sequence[Mapping[str, Any]], ranks: Sequence[Any], colors: Mapping[Any, str],
                 fields: Sequence[str], title: str, unit: str, *, reference: str | None = None) -> None:
    _time_axis(ax, title, unit)
    has_data = False
    drawn_fields: set[int] = set()
    reference_drawn = False
    for rank in ranks:
        selected = [row for row in rows if row["rank"] == rank and row["_time"] is not None]
        for index, field in enumerate(fields):
            if not any(row.get(field) is not None for row in selected):
                continue
            has_data = True
            drawn_fields.add(index)
            ax.plot(*_series(selected, field),
                    color=colors[rank], linestyle="--" if index else "-", marker="o", markersize=2.5,
                    linewidth=1.3, alpha=.9)
        if reference and any(row.get(reference) is not None for row in selected):
            reference_drawn = True
            ax.plot(*_series(selected, reference),
                    color=colors[rank], linestyle=(0, (2, 3)), linewidth=.9, alpha=.65)
    if has_data:
        ax.set_ylim(bottom=0)
        if fields == ("gpu_util_pct",):
            values = [row["gpu_util_pct"] for row in rows if row.get("gpu_util_pct") is not None]
            ax.set_ylim(0, max(105, max(values, default=100) * 1.05))
        if reference_drawn or len(fields) > 1:
            labels = ("allocated", "reserved") if len(fields) > 1 else ("used" if "mem" in fields[0] else "draw", "saved limit")
            handles = []
            if 0 in drawn_fields:
                handles.append(Line2D([], [], color=style.MUTED, label=labels[0]))
            if 1 in drawn_fields or reference_drawn:
                handles.append(Line2D([], [], color=style.MUTED, linestyle="--" if len(fields) > 1 else ":", label=labels[1]))
            ax.legend(handles=handles, loc="best", frameon=False, ncol=2, fontsize=6.5)
    else:
        style.empty(ax)


def _plot_temperature(ax: Any, rows: Sequence[Mapping[str, Any]], ranks: Sequence[Any],
                      colors: Mapping[Any, str], limit: float | None, *, rank_legend: bool = False) -> None:
    _plot_metric(ax, rows, ranks, colors, ("gpu_temp_c",), "Sampled GPU temperature", "°C")
    handles = []
    if rank_legend:
        handles = [Line2D([], [], color=colors[rank], label=f"rank {rank}") for rank in ranks
                   if any(row["rank"] == rank and row.get("gpu_temp_c") is not None and row["_time"] is not None
                          for row in rows)]
    if limit is not None:
        handles.append(ax.axhline(limit, color=style.THERMAL_STOP, linestyle=(0, (4, 3)), linewidth=1.2,
                                  label=f"Configured stop {limit:g} °C"))
    # Explicit bounds keep the stop line visible even when every reading is below it.
    values = [row["gpu_temp_c"] for row in rows
              if row["rank"] in ranks and row["_time"] is not None and row.get("gpu_temp_c") is not None]
    if limit is not None:
        values.append(limit)
    if values:
        low, high = min(0, min(values)), max(values)
        ax.set_ylim(low, high + max(5, (high - low) * .08))
    if handles:
        ax.legend(handles=handles, loc="best", frameon=False, fontsize=6.5, ncol=3 if rank_legend else 1)


def telemetry(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], aggregate: Mapping[str, Any],
              legacy: bool, ranks: Sequence[Any], colors: Mapping[Any, str], number: int) -> Any:
    name, status = _identity(config)
    fig = style.page("GPU telemetry", name, status, number, section="DEVICE DETAIL", note=_footer(legacy))
    fig.legend(handles=[Line2D([], [], color=colors[rank], linewidth=2, label=f"rank {rank}") for rank in ranks],
               loc="upper left", bbox_to_anchor=(.055, .785), ncol=6, frameon=False, fontsize=8)
    valid = {point["window"] for point in aggregate["points"]}
    plot_rows = []
    for source in rows:
        if source["rank"] not in ranks:
            continue
        row = dict(source)
        if not legacy:
            row["tflops_per_device"] = (row["total_flops"] / row["elapsed_seconds"] / 1e12
                                        if row["window"] in valid else None)
        plot_rows.append(row)
    panels = (
        (("gpu_util_pct",), "Sampled utilization", "%", None),
        (("gpu_temp_c",), "Temperature", "°C", None),
        (("gpu_mem_used_gib",), "Board memory", "GiB", "gpu_mem_total_gib"),
        (("gpu_power_w",), "Board power", "W", "gpu_power_limit_w"),
        (("cuda_peak_allocated_gib", "cuda_peak_reserved_gib"), "Allocator peaks", "GiB", None),
        (("tflops_per_device",), "Estimated device throughput", "TFLOP/s", None),
    )
    times = [row["_time"] for row in plot_rows if row["_time"] is not None]
    for index, (fields, title, unit, reference) in enumerate(panels):
        axis = fig.add_axes((.075 + .315 * (index % 3), .455 if index < 3 else .135, .24, .225))
        if fields == ("gpu_temp_c",):
            _plot_temperature(axis, plot_rows, ranks, colors, _temperature_limit(config))
        else:
            _plot_metric(axis, plot_rows, ranks, colors, fields, title, unit, reference=reference)
        if times:
            left, right = min(times), max(times)
            padding = max((right - left) * .025, dt.timedelta(seconds=.5))
            axis.set_xlim(left - padding, right + padding)
    fig.text(.075, .085, "Solid: saved readings. Dashed/dotted: allocator reserve or saved limits. Gaps remain missing; curves are not smoothed.",
             fontsize=7, color=style.MUTED)
    return fig


def comparison(devices: Sequence[Mapping[str, Any]], config: Mapping[str, Any], legacy: bool,
               colors: Mapping[Any, str], number: int) -> Any:
    name, status = _identity(config)
    fig = style.page("Device comparison", name, status, number, section="WEIGHTED RATES & SAMPLED PEAKS", note=_footer(legacy))
    columns = ((.068, "RANK", "left"), (.14, "SAVED DEVICE", "left"), (.47, "TFLOP/s", "right"),
               (.57, "UTIL %", "right"), (.66, "TEMP °C", "right"), (.76, "VRAM GiB", "right"),
               (.85, "POWER W", "right"), (.938, "UTIL DATA", "right"))
    for x, title, align in columns:
        fig.text(x, .745, title, fontsize=7.5, color=style.MUTED, weight="bold", ha=align)
    for index, item in enumerate(devices):
        y = .70 - .061 * index
        fig.add_artist(Rectangle((.055, y - .038), .89, .057, transform=fig.transFigure,
                                 facecolor="white" if index % 2 == 0 else "#eaf0f7", edgecolor="none", zorder=-1))
        fig.text(.068, y, str(item["rank"]), fontsize=10, weight="bold", color=colors[item["rank"]])
        fig.text(.14, y + .003, textwrap.shorten(item["device_name"], width=35, placeholder="…"), fontsize=8, weight="bold")
        identity = item["device_uuid"]
        fig.text(.14, y - .021, f"UUID {identity[:27]}" + ("…" if len(identity) > 27 else ""), fontsize=6.5, color=style.MUTED)
        values = (item["weighted_tflops"], item["weighted_utilization_pct"], item["max_gpu_temp_c"],
                  item["max_gpu_mem_used_gib"], item["max_gpu_power_w"])
        for (x, _, align), value in zip(columns[2:7], values):
            fig.text(x, y, "—" if value is None else _fmt(value), ha=align, fontsize=9)
        coverage = f"{item['utilization_readings']}/{item['utilization_window_count']}" if not legacy else "—"
        fig.text(.938, y, coverage, ha="right", fontsize=8, color=style.MUTED)
    if not devices:
        fig.text(.5, .55, "No saved device rows", ha="center", color=style.MUTED)
    style.note(fig, .065, .17, "Comparable, not interchangeable",
               "Per-device TFLOP/s uses summed work divided by local elapsed time. Utilization is weighted by global window duration over readable samples. Missing values are shown as a dash.", width=76)
    style.note(fig, .55, .17, "Coverage and peaks",
               "UTIL DATA is readable / accepted windows. VRAM is sampled whole-board use. Temperature and power are sampled maxima. Only accepted windows contribute; full identities are in the appendix.", width=64)
    return fig


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        return [row for key, item in value.items() for row in _flatten(item, f"{prefix}.{key}" if prefix else str(key))] or [(prefix, "{}")]
    if isinstance(value, list):
        return [row for index, item in enumerate(value) for row in _flatten(item, f"{prefix}[{index}]")] or [(prefix, "[]")]
    return [(prefix, str(value))]


def metadata(config: Mapping[str, Any], summary: Mapping[str, Any], start: int, legacy: bool) -> Iterator[Any]:
    rows = []
    for key, value in _flatten({"run": config, "summary": summary}):
        keys, values = textwrap.wrap(key, 47) or [""], textwrap.wrap(value, 100) or [""]
        height = max(len(keys), len(values))
        for offset in range(0, height, 32):
            rows.append((keys[offset:offset + 32] or ["(continued)"], values[offset:offset + 32], min(32, height - offset)))
    pages: list[list[Any]] = [[]]
    used = 0
    for row in rows:
        if used + row[2] + 1 > 39:
            pages.append([])
            used = 0
        pages[-1].append(row)
        used += row[2] + 1
    name, status = _identity(config)
    for number, rows_on_page in enumerate(pages, start):
        fig = style.page("Configuration & integrity", name, status, number, section="REPRODUCIBILITY APPENDIX", note=_footer(legacy))
        fig.text(.065, .75, "SAVED FIELD", fontsize=8, color=style.MUTED, weight="bold")
        fig.text(.40, .75, "VALUE", fontsize=8, color=style.MUTED, weight="bold")
        y = .715
        for index, (keys, values, height) in enumerate(rows_on_page):
            if index % 2 == 0:
                fig.add_artist(Rectangle((.055, y - height * .015), .89, (height + .6) * .015,
                                         transform=fig.transFigure, facecolor="white", edgecolor="none", zorder=-1))
            fig.text(.065, y, "\n".join(keys), va="top", fontsize=7, color=style.MUTED, linespacing=1.25)
            fig.text(.40, y, "\n".join(values), va="top", fontsize=7, color=style.INK, linespacing=1.25)
            y -= (height + 1) * .015
        yield fig


def figures(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], aggregate: Mapping[str, Any],
            legacy: bool, summary: Mapping[str, Any]) -> Iterator[Any]:
    ranks = _ranks(rows)
    colors = {rank: style.COLORS[index % len(style.COLORS)] for index, rank in enumerate(ranks)}
    devices = summary["per_device_summary"]
    yield overview(rows, config, aggregate, legacy, summary["temperature_summary"], colors)
    number = 2
    for offset in range(0, max(1, len(ranks)), 6):
        yield telemetry(rows, config, aggregate, legacy, ranks[offset:offset + 6], colors, number)
        number += 1
    for offset in range(0, max(1, len(devices)), 8):
        yield comparison(devices[offset:offset + 8], config, legacy, colors, number)
        number += 1
    report_config = {key: value for key, value in config.items() if key not in {"report_status", "report_error"}}
    yield from metadata(report_config, summary, number, legacy)
