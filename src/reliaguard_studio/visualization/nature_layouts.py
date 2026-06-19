from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigureSize:
    width: float
    height: float


DOUBLE_WIDE = FigureSize(12.6, 7.2)
DOUBLE_TALL = FigureSize(12.6, 8.0)
METHOD_WIDE = FigureSize(12.8, 6.9)
CONTACT_SHEET = FigureSize(18.0, 25.0)


def panel_mosaic(*rows: str) -> list[list[str]]:
    """Create a matplotlib subplot mosaic from compact strings."""
    return [list(row) for row in rows]

