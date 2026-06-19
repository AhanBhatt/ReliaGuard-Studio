from __future__ import annotations

from .figure_canvas import FigureCanvas, Panel
from .visual_identity import VI, lighten


def labeled_box(canvas: FigureCanvas, p: Panel, title: str, body: str, *, color: str) -> None:
    canvas.round_rect(p.x, p.y, p.w, p.h, fill=lighten(color, 0.86), stroke=color, radius=16, width=1.4)
    canvas.text(p.x + 16, p.y + 27, title, size=14.5, weight="800", fill=VI.ink, max_width=p.w - 26)
    canvas.text(p.x + 16, p.y + 56, body, size=11.5, fill=VI.muted, max_width=p.w - 28, line_height=1.22)
