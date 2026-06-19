from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from ..paths import PAPER_FIGURES_DIR
from .nature_layouts import CONTACT_SHEET
from .style import PALETTE, apply_nature_style


@dataclass(frozen=True)
class ContactFigure:
    number: int
    stem: str
    title: str
    score: float


def build_contact_sheet(figures: list[ContactFigure], preview_dir: Path) -> dict[str, Path]:
    apply_nature_style()
    ncols = 2
    nrows = (len(figures) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(CONTACT_SHEET.width, max(CONTACT_SHEET.height, 4.8 * nrows)))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, item in zip(axes_flat, figures, strict=False):
        path = preview_dir / f"{item.stem}.png"
        if not path.exists():
            ax.text(0.5, 0.5, f"Missing preview\n{item.stem}", ha="center", va="center")
            ax.axis("off")
            continue
        image = mpimg.imread(path)
        ax.imshow(image)
        ax.set_title(
            f"Figure {item.number}. {item.title}  |  audit {item.score:.0f}/100",
            loc="left",
            fontsize=11,
            fontweight="bold",
            color=PALETTE.ink,
            pad=8,
        )
        ax.axis("off")
    for ax in axes_flat[len(figures) :]:
        ax.axis("off")
    fig.suptitle(
        "Nature-style figure contact sheet: main manuscript figures",
        x=0.015,
        y=0.997,
        ha="left",
        fontsize=16,
        fontweight="bold",
        color=PALETTE.ink,
    )
    fig.text(
        0.015,
        0.982,
        "Scores combine file, label, source-data and reference checks; final acceptance still requires human visual review at journal proof scale.",
        ha="left",
        va="top",
        fontsize=10,
        color=PALETTE.muted,
    )
    fig.subplots_adjust(left=0.015, right=0.99, top=0.965, bottom=0.01, hspace=0.17, wspace=0.06)
    out_png = PAPER_FIGURES_DIR / "FIGURE_CONTACT_SHEET.png"
    out_pdf = PAPER_FIGURES_DIR / "FIGURE_CONTACT_SHEET.pdf"
    out_highres = PAPER_FIGURES_DIR / "FIGURE_CONTACT_SHEET_HIGHRES.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_highres, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_contact_sheet_png": out_png,
        "figure_contact_sheet_pdf": out_pdf,
        "figure_contact_sheet_highres_pdf": out_highres,
    }
