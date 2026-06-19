from __future__ import annotations

from pathlib import Path

from ..paths import PAPER_FIGURES_DIR
from .figure_canvas import FigureCanvas
from .vector_export import export_vector_canvas


def export_canvas(canvas: FigureCanvas, stem: str, *, png_dpi: int = 360) -> list[Path]:
    """Save a composed SVG figure as clean SVG, vector PDF and PNG.

    This wrapper intentionally contains no browser or browser-print fallback. If
    Cairo is unavailable, the exporter falls back to PyMuPDF SVG conversion.
    """

    return export_vector_canvas(canvas, stem, PAPER_FIGURES_DIR, png_dpi=png_dpi)
