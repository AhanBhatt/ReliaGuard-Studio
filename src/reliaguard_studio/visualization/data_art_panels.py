from __future__ import annotations

from .figure_canvas import FigureCanvas, Panel
from .visual_identity import VI


def mini_badge(canvas: FigureCanvas, p: Panel, label: str, value: str, *, color: str) -> None:
    canvas.round_rect(p.x, p.y, p.w, p.h, fill="#FFFFFF", stroke=color, radius=14, width=1.2)
    canvas.text(p.x + p.w / 2, p.y + 28, value, size=20, fill=color, weight="800", anchor="middle")
    canvas.text(p.x + p.w / 2, p.y + 52, label, size=10.5, fill=VI.muted, anchor="middle", max_width=p.w - 16)
