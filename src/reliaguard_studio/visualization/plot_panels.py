from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from .figure_canvas import FigureCanvas, Panel
from .visual_identity import VI, lighten


def _finite(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values if pd.notna(v) and math.isfinite(float(v))]


def _scale(value: float, lo: float, hi: float, start: float, length: float) -> float:
    if hi == lo:
        return start + length / 2
    return start + (float(value) - lo) / (hi - lo) * length


def horizontal_interval_plot(
    canvas: FigureCanvas,
    p: Panel,
    frame: pd.DataFrame,
    *,
    label_col: str,
    estimate_col: str,
    low_col: str,
    high_col: str,
    color: str,
    xlabel: str,
    zero: float = 0.0,
    xlim: tuple[float, float] | None = None,
    value_fmt: str = "{:.2f}",
) -> None:
    if frame.empty:
        canvas.text(p.x + p.w / 2, p.y + p.h / 2, "No estimable contrast", size=16, fill=VI.muted, anchor="middle")
        return
    frame = frame.reset_index(drop=True)
    all_values = _finite(frame[estimate_col].tolist() + frame[low_col].tolist() + frame[high_col].tolist() + [zero])
    lo, hi = xlim or (min(all_values), max(all_values))
    pad = (hi - lo) * 0.08 or 0.1
    if xlim is None:
        lo -= pad
        hi += pad
    left = p.x + 220
    top = p.y + 42
    plot_w = p.w - 290
    plot_h = p.h - 92
    row_h = plot_h / max(len(frame), 1)
    canvas.axis(left, top, plot_w, plot_h, xlabel=xlabel)
    zx = _scale(zero, lo, hi, left, plot_w)
    canvas.line(zx, top, zx, top + plot_h, stroke=VI.ink, width=1.0, opacity=0.45)
    for idx, row in frame.iterrows():
        y = top + row_h * (idx + 0.5)
        label = str(row[label_col])
        est = float(row[estimate_col])
        low = float(row[low_col])
        high = float(row[high_col])
        lx = _scale(low, lo, hi, left, plot_w)
        hx = _scale(high, lo, hi, left, plot_w)
        ex = _scale(est, lo, hi, left, plot_w)
        canvas.text(p.x + 18, y + 5, label, size=12.2, fill=VI.ink, max_width=190)
        canvas.line(lx, y, hx, y, stroke=lighten(color, 0.25), width=4.0, opacity=0.95)
        canvas.dot(ex, y, 7, fill=color)
        canvas.text(min(p.x + p.w - 14, hx + 12), y + 5, value_fmt.format(est), size=11.2, fill=VI.muted)


def grouped_bar_ci(
    canvas: FigureCanvas,
    p: Panel,
    frame: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    low_col: str | None,
    high_col: str | None,
    color_col: str | None = None,
    colors: dict[str, str] | None = None,
    ylabel: str = "",
    ylim: tuple[float, float] | None = None,
    horizontal: bool = False,
) -> None:
    if frame.empty:
        canvas.text(p.x + p.w / 2, p.y + p.h / 2, "No data", size=16, fill=VI.muted, anchor="middle")
        return
    frame = frame.reset_index(drop=True)
    vals = [float(v) for v in frame[value_col]]
    lo, hi = ylim or (min(0, min(vals)), max(vals) * 1.18 + 0.02)
    if horizontal:
        left = p.x + 205
        top = p.y + 35
        plot_w = p.w - 245
        plot_h = p.h - 76
        row_h = plot_h / len(frame)
        canvas.axis(left, top, plot_w, plot_h, xlabel=ylabel)
        for idx, row in frame.iterrows():
            y = top + row_h * idx + row_h * 0.25
            value = float(row[value_col])
            x0 = _scale(0, lo, hi, left, plot_w)
            x1 = _scale(value, lo, hi, left, plot_w)
            color = colors.get(str(row[color_col]), VI.accent) if colors and color_col else VI.accent
            canvas.text(p.x + 16, y + row_h * 0.45, str(row[label_col]), size=12, fill=VI.ink, max_width=180)
            canvas.round_rect(min(x0, x1), y, abs(x1 - x0), row_h * 0.50, fill=color, stroke="none", radius=5)
            if low_col and high_col:
                lx = _scale(float(row[low_col]), lo, hi, left, plot_w)
                hx = _scale(float(row[high_col]), lo, hi, left, plot_w)
                canvas.line(lx, y + row_h * 0.25, hx, y + row_h * 0.25, stroke=VI.ink, width=1.3)
            canvas.text(x1 + 8, y + row_h * 0.38, f"{value:.2f}", size=10.5, fill=VI.muted)
        return
    left = p.x + 58
    top = p.y + 40
    plot_w = p.w - 90
    plot_h = p.h - 96
    canvas.axis(left, top, plot_w, plot_h, ylabel=ylabel)
    bar_w = plot_w / len(frame) * 0.58
    for idx, row in frame.iterrows():
        cx = left + plot_w * (idx + 0.5) / len(frame)
        value = float(row[value_col])
        y0 = _scale(0, lo, hi, top + plot_h, -plot_h)
        y1 = _scale(value, lo, hi, top + plot_h, -plot_h)
        color = colors.get(str(row[color_col]), VI.accent) if colors and color_col else VI.accent
        canvas.round_rect(cx - bar_w / 2, min(y0, y1), bar_w, abs(y1 - y0), fill=color, stroke="none", radius=5)
        if low_col and high_col:
            ly = _scale(float(row[low_col]), lo, hi, top + plot_h, -plot_h)
            hy = _scale(float(row[high_col]), lo, hi, top + plot_h, -plot_h)
            canvas.line(cx, ly, cx, hy, stroke=VI.ink, width=1.2)
            canvas.line(cx - 7, ly, cx + 7, ly, stroke=VI.ink, width=1.2)
            canvas.line(cx - 7, hy, cx + 7, hy, stroke=VI.ink, width=1.2)
        canvas.text(cx, top + plot_h + 24, str(row[label_col]), size=10.8, fill=VI.ink, anchor="middle", max_width=92)
        canvas.text(cx, y1 - 8 if y1 < y0 else y1 + 16, f"{value:.2f}", size=10.4, fill=VI.muted, anchor="middle")


def reliability_plot(
    canvas: FigureCanvas,
    p: Panel,
    frame: pd.DataFrame,
    *,
    model_col: str,
    x_col: str,
    y_col: str,
    models: list[str],
    colors: dict[str, str],
    title: str | None = None,
) -> None:
    left = p.x + 54
    top = p.y + 42
    plot_w = p.w - 86
    plot_h = p.h - 92
    canvas.axis(left, top, plot_w, plot_h, xlabel="predicted risk", ylabel="empirical rate")
    if title:
        canvas.text(p.x + 16, p.y + 28, title, size=13.5, weight="700")
    canvas.line(left, top + plot_h, left + plot_w, top, stroke="#A8B4BC", width=1.3, opacity=0.65)
    for idx, model in enumerate(models):
        sub = frame.loc[frame[model_col].eq(model)].sort_values(x_col)
        if sub.empty:
            continue
        color = colors.get(model, VI.accent)
        points: list[tuple[float, float]] = []
        for _, row in sub.iterrows():
            x = _scale(float(row[x_col]), 0, 1, left, plot_w)
            y = _scale(float(row[y_col]), 0, 1, top + plot_h, -plot_h)
            points.append((x, y))
        if len(points) > 1:
            canvas.dwg.add(canvas.dwg.polyline(points=points, fill="none", stroke=color, stroke_width=3.0, opacity=0.95))
        for x, y in points:
            canvas.dot(x, y, 5, fill=color)
        if points:
            canvas.text(points[-1][0] + 7, points[-1][1] + 4, model, size=10.5, fill=color, max_width=125)


def matrix_plot(
    canvas: FigureCanvas,
    p: Panel,
    matrix: pd.DataFrame,
    *,
    row_labels: list[str],
    col_labels: list[str],
    value_fmt: str = "{:.2f}",
    lo: float = 0.45,
    hi: float = 0.85,
) -> None:
    if matrix.empty:
        canvas.text(p.x + p.w / 2, p.y + p.h / 2, "No transfer matrix", size=16, fill=VI.muted, anchor="middle")
        return
    left = p.x + 145
    top = p.y + 56
    cell_w = (p.w - 170) / max(len(col_labels), 1)
    cell_h = (p.h - 92) / max(len(row_labels), 1)
    canvas.text(left + (cell_w * len(col_labels)) / 2, p.y + 28, "Test dataset", size=12.5, fill=VI.muted, anchor="middle")
    for j, label in enumerate(col_labels):
        canvas.text(left + j * cell_w + cell_w / 2, top - 15, label, size=11, weight="700", anchor="middle", max_width=cell_w - 8)
    for i, label in enumerate(row_labels):
        canvas.text(p.x + 14, top + i * cell_h + cell_h / 2 + 5, label, size=11, fill=VI.ink, max_width=120)
        for j, col in enumerate(matrix.columns):
            value = matrix.iloc[i, j]
            frac = 0 if pd.isna(value) else max(0, min(1, (float(value) - lo) / (hi - lo)))
            color = blend("#EEF6F8", "#0B7285", frac)
            x = left + j * cell_w
            y = top + i * cell_h
            canvas.round_rect(x + 3, y + 3, cell_w - 6, cell_h - 6, fill=color, stroke="#D6DEE4", radius=9)
            label_text = "NA" if pd.isna(value) else value_fmt.format(float(value))
            canvas.text(x + cell_w / 2, y + cell_h / 2 + 5, label_text, size=12, weight="700", anchor="middle", fill=VI.ink)


def blend(color_a: str, color_b: str, amount: float) -> str:
    a = color_a.lstrip("#")
    b = color_b.lstrip("#")
    ra = [int(a[i : i + 2], 16) for i in (0, 2, 4)]
    rb = [int(b[i : i + 2], 16) for i in (0, 2, 4)]
    vals = [round(x + (y - x) * amount) for x, y in zip(ra, rb, strict=False)]
    return "#" + "".join(f"{v:02x}" for v in vals)
