from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re

import matplotlib.image as mpimg

from ..paths import PAPER_SOURCE_DATA_DIR


FORBIDDEN_MAIN_FIGURE_TERMS = {
    "delayed recall",
    "long-term transfer",
    "cognitive decline",
    "clinical diagnosis",
    "dataset_name",
}

FORBIDDEN_VISIBLE_TOKENS = [
    "dataset_name",
    "chi2023_dke",
    "convxai_iui2025",
    "pardos_chatgpt_tutoring",
    "flora_ips",
    "Undrreliance",
    "Ai",
    "Xai",
    "Llm",
]

ALLOWED_SNAKE_TOKENS = {"h0", "h1"}


@dataclass(frozen=True)
class FigureSpec:
    number: int
    stem: str
    title: str
    min_width_px: int = 1800
    min_height_px: int = 1200
    expected_panels: str = "ABCD"


def image_dimensions(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    image = mpimg.imread(path)
    return int(image.shape[1]), int(image.shape[0])


def svg_text_nodes(path: Path) -> list[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="ignore")
    return [html.unescape(match).strip() for match in re.findall(r"<text[^>]*>(.*?)</text>", raw, flags=re.DOTALL) if match.strip()]


def visible_text_issues(svg: Path) -> list[str]:
    text = "\n".join(svg_text_nodes(svg))
    issues: list[str] = []
    for token in FORBIDDEN_VISIBLE_TOKENS:
        if token in text:
            issues.append(f"forbidden visible token: {token}")
    if re.search(r"\bxai\b", text):
        issues.append("lowercase XAI token")
    snake_tokens = [token for token in re.findall(r"\b[a-z]+_[a-z0-9_]+\b", text) if token not in ALLOWED_SNAKE_TOKENS]
    if snake_tokens:
        issues.append(f"raw snake-case label: {snake_tokens[0]}")
    return issues


def score_figure(spec: FigureSpec, latex: str, audit_text: str, figures_dir: Path) -> tuple[float, list[str]]:
    issues: list[str] = []
    pdf = figures_dir / f"{spec.stem}.pdf"
    svg = figures_dir / f"{spec.stem}.svg"
    png = figures_dir / f"{spec.stem}.png"
    if not pdf.exists():
        issues.append("missing PDF")
    if not svg.exists():
        issues.append("missing SVG")
    if not png.exists():
        issues.append("missing PNG")
    width, height = image_dimensions(png)
    if width < spec.min_width_px or height < spec.min_height_px:
        issues.append(f"preview below target size ({width}x{height})")
    if f"{spec.stem}.pdf" not in latex:
        issues.append("not referenced in LaTeX")
    if "\\caption{" not in latex[latex.find(f"{spec.stem}.pdf") : latex.find(f"{spec.stem}.pdf") + 1200]:
        issues.append("caption not detected near figure reference")
    if spec.stem not in audit_text:
        issues.append("not listed in figure audit")
    panel_text = "".join(svg_text_nodes(svg))
    for panel in spec.expected_panels:
        if panel not in panel_text:
            issues.append(f"missing expected panel label {panel}")
    issues.extend(visible_text_issues(svg))
    if not list(PAPER_SOURCE_DATA_DIR.glob(f"{spec.stem}_*.*")):
        issues.append("missing source data")
    lower_latex = latex.lower()
    for term in FORBIDDEN_MAIN_FIGURE_TERMS:
        if term in lower_latex and "unsupported" not in lower_latex:
            issues.append(f"forbidden or unsupported term appears without boundary: {term}")
    score = 100.0
    penalties = {
        "missing PDF": 25.0,
        "missing SVG": 20.0,
        "missing PNG": 20.0,
        "not referenced in LaTeX": 20.0,
        "caption not detected near figure reference": 10.0,
        "not listed in figure audit": 10.0,
        "missing source data": 20.0,
    }
    for issue in issues:
        score -= penalties.get(issue, 5.0)
    return max(score, 0.0), issues
