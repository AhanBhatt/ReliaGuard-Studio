from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..paths import PAPER_FIGURES_DIR


def export_figure(stem: str, *, dpi: int = 360) -> list[Path]:
    """Export the active figure in vector-first publication formats."""
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        PAPER_FIGURES_DIR / f"{stem}.pdf",
        PAPER_FIGURES_DIR / f"{stem}.svg",
        PAPER_FIGURES_DIR / f"{stem}.png",
    ]
    fig = plt.gcf()
    fig.set_facecolor("white")
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    fig.savefig(outputs[2], dpi=dpi, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return outputs

