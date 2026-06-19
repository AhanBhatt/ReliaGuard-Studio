from __future__ import annotations

from .figure_canvas import FigureCanvas
from .visual_identity import VI, lighten


def flow_node(canvas: FigureCanvas, x: float, y: float, w: float, h: float, label: str, *, color: str) -> tuple[float, float]:
    canvas.round_rect(x, y, w, h, fill=lighten(color, 0.84), stroke=color, radius=14, width=1.3)
    canvas.text(x + w / 2, y + h / 2 + 5, label, size=12.2, fill=VI.ink, weight="700", anchor="middle", max_width=w - 16)
    return x + w, y + h / 2
