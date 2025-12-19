"""
PDF reporting utilities for GPU stress test runs.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import torch
from matplotlib.backends.backend_pdf import PdfPages

A4_SIZE = (8.27, 11.69)  # inches
DPI = 300

def _estimate_cuda_cores_per_sm(major: int, minor: int) -> int:
    # Approximate cores per SM by architecture family.
    if major >= 8:
        return 128  # Ampere/Hopper-ish
    if major == 7:
        return 64   # Volta/Turing
    if major == 6:
        return 128  # Pascal
    if major == 5:
        return 128  # Maxwell
    return 64


def gather_gpu_info() -> List[Dict[str, str]]:
    gpus: List[Dict[str, str]] = []
    smi_info: Dict[int, Dict[str, str]] = {}
    # Try to gather extended info via nvidia-smi.
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                idx = int(parts[0])
                smi_info[idx] = {
                    "name": parts[1],
                    "memory_total_gb": f"{float(parts[2]) / 1024:.2f}",
                    "power_limit_w": parts[3],
                }
    except Exception:
        pass

    if not torch.cuda.is_available():
        return gpus

    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        cores_per_sm = _estimate_cuda_cores_per_sm(props.major, props.minor)
        approx_cores = props.multi_processor_count * cores_per_sm
        entry = {
            "gpu_id": str(idx),
            "name": props.name,
            "device_type": "CUDA",
            "total_memory_gb": f"{props.total_memory / (1024**3):.2f}",
            "sm_count": str(props.multi_processor_count),
            "approx_cuda_cores": str(approx_cores),
            "compute_capability": f"{props.major}.{props.minor}",
            "power_limit_w": smi_info.get(idx, {}).get("power_limit_w", "n/a"),
        }
        # Override name/memory from nvidia-smi if available
        if idx in smi_info:
            entry["name"] = smi_info[idx].get("name", entry["name"])
            entry["total_memory_gb"] = smi_info[idx].get("memory_total_gb", entry["total_memory_gb"])
        gpus.append(entry)
    return gpus


def _load_metrics(metrics_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _load_config(config_path: Path) -> Dict[str, str]:
    with config_path.open() as f:
        return json.load(f)


def _add_table(fig, title: str, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    ax = fig.add_subplot(111)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center")
    table.scale(1, 1.2)
    ax.set_title(title, fontsize=12, pad=10)


def _plot_metric_multi(ax, series: Dict[str, Dict[str, List]], title: str) -> None:
    if not series:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    for label, data in series.items():
        if not data["ts"] or not data["vals"]:
            continue
        ax.plot(data["ts"], data["vals"], label=label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")


def generate_report(metrics_path: Path, config_path: Path, output_pdf: Path) -> None:
    metrics_rows = _load_metrics(metrics_path)
    cfg = _load_config(config_path)
    gpus = gather_gpu_info()

    created_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def extract_series_by_gpu(key: str) -> Dict[str, Dict[str, List]]:
        series: Dict[str, Dict[str, List]] = {}
        for r in metrics_rows:
            gpu_label = f"{r.get('gpu_id','n/a')} - {r.get('gpu_name','')}"
            v = r.get(key)
            if v in ("n/a", "", None):
                continue
            series.setdefault(gpu_label, {"ts": [], "vals": []})
            series[gpu_label]["ts"].append(dt.datetime.fromisoformat(r["timestamp"]))
            series[gpu_label]["vals"].append(float(v))
        return series

    tflops_total = extract_series_by_gpu("tflops_total")
    tflops_per_gpu = extract_series_by_gpu("tflops_per_gpu")
    temps = extract_series_by_gpu("gpu_temp_c")
    utils = extract_series_by_gpu("gpu_util_pct")
    mem_gb = extract_series_by_gpu("gpu_mem_used_gb")
    mem_pct = extract_series_by_gpu("gpu_mem_used_pct")
    power_w = extract_series_by_gpu("gpu_power_w")
    power_pct = extract_series_by_gpu("gpu_power_pct")

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_pdf) as pdf:
        # GPU summary table
        # Cover page / header
        fig = plt.figure(figsize=A4_SIZE, dpi=DPI)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.8, "GPU Stress Test Report", fontsize=18, ha="center", va="center", weight="bold")
        ax.text(0.5, 0.7, f"Generated: {created_at}", fontsize=12, ha="center", va="center")
        ax.text(0.5, 0.6, f"Run: {metrics_path.parent.name}", fontsize=12, ha="center", va="center")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if gpus:
            fig = plt.figure(figsize=A4_SIZE, dpi=DPI)
            columns = ["GPU ID", "Name", "Type", "VRAM (GB)", "SMs", "Approx CUDA Cores", "Compute Cap", "Power Limit (W)"]
            rows = [
                [
                    g["gpu_id"],
                    g["name"],
                    g["device_type"],
                    g["total_memory_gb"],
                    g["sm_count"],
                    g["approx_cuda_cores"],
                    g["compute_capability"],
                    g["power_limit_w"],
                ]
                for g in gpus
            ]
            _add_table(fig, "GPU Summary", columns, rows)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Model/config table
        fig = plt.figure(figsize=A4_SIZE, dpi=DPI)
        columns = ["Field", "Value"]
        keys = [
            "batch_size",
            "num_batches",
            "num_epochs",
            "input_dim",
            "hidden_dim",
            "output_dim",
            "num_hidden_layers",
            "extra_matmul_repeats",
            "extra_matmul_dim",
            "generate_data_on_device",
            "log_every_n_steps",
        ]
        rows = [[k, str(cfg.get(k, "n/a"))] for k in keys]
        _add_table(fig, "Model / Run Config", columns, rows)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Metric plots
        fig, axes = plt.subplots(3, 2, figsize=A4_SIZE, dpi=DPI, sharex=True)
        ax_list = axes.flatten()
        series = [
            ("TFLOPS Total", tflops_total),
            ("TFLOPS per GPU", tflops_per_gpu),
            ("GPU Util (%)", utils),
            ("GPU Temp (C)", temps),
            ("Memory Used (GB)", mem_gb),
            ("Power (W)", power_w),
        ]
        for ax, (label, data) in zip(ax_list, series):
            _plot_metric_multi(ax, data, label)
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Power percent plot removed per request
