from __future__ import annotations

from pathlib import Path

from .figure_canvas import FigureCanvas
from .figure_export import export_canvas


def save_composed_figure(canvas: FigureCanvas, stem: str) -> list[Path]:
    return export_canvas(canvas, stem)
