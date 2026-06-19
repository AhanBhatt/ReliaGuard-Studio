from __future__ import annotations

import matplotlib.pyplot as plt

from .style import PALETTE


def add_panel_label(ax: plt.Axes, label: str, title: str | None = None) -> None:
    ax.text(
        -0.14,
        1.075,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        color=PALETTE.ink,
    )
    if title:
        ax.set_title(title, loc="left", fontweight="bold", color=PALETTE.ink, pad=9)


def direct_label(ax: plt.Axes, x: float, y: float, text: str, color: str, *, ha: str = "left") -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        ha=ha,
        va="center",
        fontsize=8.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": color, "linewidth": 0.7, "alpha": 0.92},
    )
