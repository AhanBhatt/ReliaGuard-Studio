from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..paths import PAPER_DIR, PAPER_FIGURES_DIR, PAPER_SOURCE_DATA_DIR


@dataclass(frozen=True)
class RegistryFigure:
    number: int
    stem: str
    title: str
    expected_panels: str
    latex_label: str
    caption_title: str
    source_data_glob: str

    @property
    def svg_path(self) -> Path:
        return PAPER_FIGURES_DIR / f"{self.stem}.svg"

    @property
    def pdf_path(self) -> Path:
        return PAPER_FIGURES_DIR / f"{self.stem}.pdf"

    @property
    def png_path(self) -> Path:
        return PAPER_FIGURES_DIR / f"{self.stem}.png"

    @property
    def metadata_path(self) -> Path:
        return PAPER_FIGURES_DIR / "metadata" / f"figure_{self.number:02d}.json"

    @property
    def source_data_paths(self) -> list[Path]:
        return sorted(PAPER_SOURCE_DATA_DIR.glob(self.source_data_glob))


def registry_path() -> Path:
    return PAPER_DIR / "figure_registry.yaml"


def load_figure_registry(path: Path | None = None) -> list[RegistryFigure]:
    raw = yaml.safe_load((path or registry_path()).read_text(encoding="utf-8"))
    return [RegistryFigure(**item) for item in raw.get("figures", [])]


def figure_by_number(number: int) -> RegistryFigure:
    for figure in load_figure_registry():
        if figure.number == number:
            return figure
    raise KeyError(f"No registered figure number {number}")


def figure_by_stem(stem: str) -> RegistryFigure:
    for figure in load_figure_registry():
        if figure.stem == stem:
            return figure
    raise KeyError(f"No registered figure stem {stem}")
