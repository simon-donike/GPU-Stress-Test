"""Shared print-friendly design system for saved performance reports."""
from __future__ import annotations

import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

PAGE_SIZE = (11.69, 8.27)  # ISO A4, landscape; every PDF page uses the same size.
DPI = 180
INK = "#14243a"
MUTED = "#53657c"
PAPER = "#f3f6fb"
GRID = "#dfe7f0"
ACCENT = "#22d3ee"
THERMAL_STOP = "#b33c4c"
COLORS = ("#2463ce", "#c56a12", "#16816a", "#8b4cc4", "#c43e67", "#087e99", "#687338", "#596780")
RC = {
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.facecolor": "white", "axes.labelcolor": MUTED,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.labelsize": 8,
    "legend.fontsize": 7, "lines.linewidth": 1.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": PAPER,
    "pdf.use14corefonts": False, "text.usetex": False,
    "text.parse_math": False,
    "savefig.bbox": None, "savefig.transparent": False,
    "axes.formatter.use_locale": False, "timezone": "UTC",
}


def page(title: str, run_name: str, status: str, number: int, *, section: str = "RUN REPORT",
         note: str = "Analytical matmul estimates · sampled GPU telemetry · UTC timestamps") -> Any:
    fig = plt.figure(figsize=PAGE_SIZE, dpi=DPI, facecolor=PAPER)
    fig.add_artist(Rectangle((0, .855), 1, .145, transform=fig.transFigure, color=INK, zorder=-1))
    fig.add_artist(Rectangle((.055, .855), .095, .004, transform=fig.transFigure, color=ACCENT))
    fig.text(.055, .966, f"GPU STRESS TEST   /   {section}", fontsize=8, color="#a9bfd8", weight="bold")
    fig.text(.055, .895, title, fontsize=24, weight="bold", color="white")
    state = str(status).replace("_", " ").upper()
    badge = "#175f52" if state == "COMPLETED" else "#6d4b17" if state in {"RUNNING", "UNKNOWN"} else "#74364b"
    fig.text(.945, .955, state[:32], ha="right", va="center", fontsize=8, color="white",
             bbox={"boxstyle": "round,pad=0.5", "facecolor": badge, "edgecolor": "none"})
    fig.text(.055, .818, textwrap.fill(str(run_name), 115), fontsize=9, color=MUTED, va="top")
    fig.add_artist(Line2D([.055, .945], [.064, .064], transform=fig.transFigure, color=GRID, linewidth=.8))
    fig.text(.055, .034, note, fontsize=7, color=MUTED)
    fig.text(.945, .034, f"{number:02d}", ha="right", fontsize=9, color=INK, weight="bold")
    return fig


def card(fig: Any, x: float, y: float, width: float, label: str, value: str, detail: str,
         *, color: str = COLORS[0], height: float = .12) -> None:
    fig.add_artist(FancyBboxPatch((x, y), width, height, transform=fig.transFigure,
                                 boxstyle="round,pad=0.006,rounding_size=0.009",
                                 facecolor="white", edgecolor=GRID, linewidth=.7, zorder=-1))
    fig.text(x + .014, y + height - .027, label.upper(), fontsize=7.5, color=MUTED, weight="bold")
    value_size = min(21, (width - .028) * PAGE_SIZE[0] * 72 / max(1, len(value) * .63))
    fig.text(x + .014, y + height - .072, value, fontsize=value_size, weight="bold", color=color)
    fig.text(x + .014, y + .014, detail, fontsize=7, color=MUTED)


def axis(ax: Any, title: str, unit: str) -> None:
    ax.set_title(title, loc="left", pad=10, color=INK)
    ax.set_ylabel(unit, labelpad=7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(axis="both", labelsize=7, length=3, color=GRID)
    ax.yaxis.get_offset_text().set_fontsize(7)
    ax.margins(x=.035)


def empty(ax: Any, message: str = "No saved readings") -> None:
    ax.text(.5, .5, message, transform=ax.transAxes, va="center", ha="center", fontsize=9, color=MUTED,
            bbox={"boxstyle": "round,pad=0.65", "facecolor": PAPER, "edgecolor": "none"})
    ax.set_xticks([])
    ax.set_yticks([])


def note(fig: Any, x: float, y: float, title: str, body: str, *, width: int = 62) -> None:
    fig.text(x, y, title.upper(), fontsize=8, weight="bold", color=INK, va="top")
    fig.text(x, y - .030, textwrap.fill(body, width), fontsize=8, color=MUTED, va="top", linespacing=1.45)
