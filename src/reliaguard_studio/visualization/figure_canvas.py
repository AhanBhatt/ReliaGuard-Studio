from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import svgwrite

from .visual_identity import VI, lighten


@dataclass(frozen=True)
class Panel:
    x: float
    y: float
    w: float
    h: float

    def inset(self, dx: float = 18, dy: float = 18) -> "Panel":
        return Panel(self.x + dx, self.y + dy, self.w - 2 * dx, self.h - 2 * dy)


class FigureCanvas:
    """Small SVG-native layout helper for publication figures."""

    def __init__(self, width: int = 1800, height: int = 1200, title: str | None = None) -> None:
        self.width = width
        self.height = height
        self.dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"), profile="full")
        self.dwg.viewbox(0, 0, width, height)
        self.dwg.add(self.dwg.rect(insert=(0, 0), size=(width, height), fill=VI.white))
        if title:
            self.text(42, 46, title, size=30, weight="700", fill=VI.ink)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.dwg.saveas(str(path))

    def panel(self, label: str, x: float, y: float, w: float, h: float, title: str | None = None) -> Panel:
        p = Panel(x, y, w, h)
        self.round_rect(x, y, w, h, fill=VI.panel_bg, stroke="#D6DEE4", radius=18, opacity=1.0)
        self.text(x + 16, y + 30, label, size=22, weight="800", fill=VI.ink)
        if title:
            self.text(x + 52, y + 30, title, size=18, weight="700", fill=VI.ink)
        return p

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: float = 14,
        fill: str = VI.ink,
        weight: str = "400",
        anchor: str = "start",
        max_width: float | None = None,
        line_height: float = 1.22,
        opacity: float = 1.0,
    ) -> None:
        if max_width is None:
            self.dwg.add(
                self.dwg.text(
                    str(text),
                    insert=(x, y),
                    fill=fill,
                    font_size=size,
                    font_family=VI.font,
                    font_weight=weight,
                    text_anchor=anchor,
                    opacity=opacity,
                )
            )
            return
        chars = max(8, int(max_width / (size * 0.52)))
        lines = textwrap.wrap(text, width=chars, break_long_words=False)
        for idx, line in enumerate(lines):
            self.dwg.add(
                self.dwg.text(
                    str(line),
                    insert=(x, y + idx * size * line_height),
                    fill=fill,
                    font_size=size,
                    font_family=VI.font,
                    font_weight=weight,
                    text_anchor=anchor,
                    opacity=opacity,
                )
            )

    def round_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str,
        stroke: str = "none",
        radius: float = 10,
        width: float = 1,
        opacity: float = 1,
    ) -> None:
        self.dwg.add(
            self.dwg.rect(
                insert=(x, y),
                size=(w, h),
                rx=radius,
                ry=radius,
                fill=fill,
                stroke=stroke,
                stroke_width=width,
                opacity=opacity,
            )
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = VI.muted, width: float = 1.2, opacity: float = 1) -> None:
        self.dwg.add(self.dwg.line(start=(x1, y1), end=(x2, y2), stroke=stroke, stroke_width=width, opacity=opacity))

    def arrow(self, x1: float, y1: float, x2: float, y2: float, *, stroke: str = VI.muted, width: float = 1.6) -> None:
        marker_id = f"arrow_{abs(hash((x1, y1, x2, y2, stroke))) % 1000000}"
        marker = self.dwg.marker(id=marker_id, insert=(10, 5), size=(10, 10), orient="auto", markerUnits="strokeWidth")
        marker.add(self.dwg.path(d="M 0 0 L 10 5 L 0 10 z", fill=stroke))
        self.dwg.defs.add(marker)
        self.dwg.add(
            self.dwg.line(
                start=(x1, y1),
                end=(x2, y2),
                stroke=stroke,
                stroke_width=width,
                marker_end=f"url(#{marker_id})",
            )
        )

    def dot(self, x: float, y: float, r: float, *, fill: str, stroke: str = "white", width: float = 1.0) -> None:
        self.dwg.add(self.dwg.circle(center=(x, y), r=r, fill=fill, stroke=stroke, stroke_width=width))

    def pill(self, x: float, y: float, text: str, *, fill: str, stroke: str | None = None, text_fill: str = VI.ink, w: float | None = None) -> None:
        w = w or max(92, len(text) * 7.2 + 24)
        self.round_rect(x, y, w, 30, fill=fill, stroke=stroke or lighten(fill, 0.25), radius=15, width=0.8)
        self.text(x + w / 2, y + 20, text, size=12, fill=text_fill, weight="700", anchor="middle")

    def axis(self, x: float, y: float, w: float, h: float, *, xlabel: str = "", ylabel: str = "") -> None:
        self.line(x, y + h, x + w, y + h, stroke="#9AA8B2", width=1.2)
        self.line(x, y, x, y + h, stroke="#9AA8B2", width=1.2)
        if xlabel:
            self.text(x + w / 2, y + h + 34, xlabel, size=13, fill=VI.muted, anchor="middle")
        if ylabel:
            self.text(x - 36, y + h / 2, ylabel, size=13, fill=VI.muted, anchor="middle")

    def card(self, p: Panel, title: str, body: str, *, color: str, icon: str = "") -> None:
        self.round_rect(p.x, p.y, p.w, p.h, fill=lighten(color, 0.88), stroke=color, radius=18, width=1.4)
        if icon:
            self.text(p.x + 18, p.y + 31, icon, size=20, fill=color, weight="800")
            title_x = p.x + 48
        else:
            title_x = p.x + 18
        self.text(title_x, p.y + 30, title, size=16, fill=VI.ink, weight="800", max_width=p.w - (title_x - p.x) - 16)
        self.text(p.x + 18, p.y + 62, body, size=12.4, fill=VI.muted, max_width=p.w - 32, line_height=1.25)


def grid(panel: Panel, rows: int, cols: int, *, gap_x: float = 16, gap_y: float = 16) -> list[Panel]:
    cell_w = (panel.w - gap_x * (cols - 1)) / cols
    cell_h = (panel.h - gap_y * (rows - 1)) / rows
    return [
        Panel(panel.x + c * (cell_w + gap_x), panel.y + r * (cell_h + gap_y), cell_w, cell_h)
        for r in range(rows)
        for c in range(cols)
    ]
