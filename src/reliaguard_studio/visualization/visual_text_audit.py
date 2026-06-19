from __future__ import annotations

from dataclasses import dataclass
import html
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from ..paths import PAPER_DIR, PAPER_FIGURES_DIR, REPO_ROOT
from .figure_registry import load_figure_registry


BAD_LITERAL_TOKENS = [
    "file:///",
    "D:/",
    "C:/",
    "Users/",
    "Projects/NeuroSymbolic",
    "dataset_name",
    "Undrreliance",
    "chi2023_dke",
    "convxai_iui2025",
    "pardos_chatgpt_tutoring",
    "flora_ips",
    "snake_case",
]

BAD_REGEX_TOKENS = {
    "browser page counter": re.compile(r"(?<![\d.])1/1(?![\d.])"),
    "raw reliance suffix": re.compile(r"(?:\\_|\b_)reliance\b"),
    "bad AI capitalization": re.compile(r"\bAi\b"),
    "bad XAI capitalization": re.compile(r"\bXai\b"),
    "bad LLM capitalization": re.compile(r"\bLlm\b"),
    "raw snake-case label": re.compile(r"\b(?!h0\b|h1\b)[a-z]+_[a-z0-9_]+\b"),
}


@dataclass(frozen=True)
class TextFinding:
    artifact: str
    source_type: str
    token: str
    excerpt: str


def _normalise_text(text: str) -> str:
    text = html.unescape(text)
    return text.replace("\\_", "_")


def _extract_svg_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8", errors="ignore"))
    except ET.ParseError:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"<text[^>]*>(.*?)</text>", raw, flags=re.DOTALL)
        return "\n".join(_normalise_text(re.sub(r"<[^>]+>", "", match)) for match in matches)
    pieces: list[str] = []
    for node in root.iter():
        if node.tag.endswith("text") or node.tag.endswith("tspan"):
            if node.text and node.text.strip():
                pieces.append(_normalise_text(node.text.strip()))
    return "\n".join(pieces)


def _extract_pdf_text(path: Path) -> str:
    if not path.exists():
        return ""
    import fitz

    document = fitz.open(str(path))
    try:
        return "\n".join(page.get_text("text") for page in document)
    finally:
        document.close()


def _extract_text_file(path: Path) -> str:
    if not path.exists():
        return ""
    return _normalise_text(path.read_text(encoding="utf-8", errors="ignore"))


def _context(text: str, start: int, end: int) -> str:
    lo = max(0, start - 45)
    hi = min(len(text), end + 45)
    return " ".join(text[lo:hi].split())


def _scan_text(path: Path, source_type: str, text: str, *, allow_svg_filename: bool = False) -> list[TextFinding]:
    findings: list[TextFinding] = []
    text = _normalise_text(text)
    for token in BAD_LITERAL_TOKENS:
        start = text.find(token)
        if start >= 0:
            findings.append(TextFinding(str(path.relative_to(REPO_ROOT)), source_type, token, _context(text, start, start + len(token))))
    for label, pattern in BAD_REGEX_TOKENS.items():
        match = pattern.search(text)
        if match:
            findings.append(TextFinding(str(path.relative_to(REPO_ROOT)), source_type, label, _context(text, match.start(), match.end())))
    if not allow_svg_filename:
        match = re.search(r"\b[\w.-]+\.svg\b", text, flags=re.IGNORECASE)
        if match:
            findings.append(TextFinding(str(path.relative_to(REPO_ROOT)), source_type, ".svg visible text", _context(text, match.start(), match.end())))
    return findings


def _iter_main_figure_artifacts() -> list[tuple[Path, str]]:
    artifacts: list[tuple[Path, str]] = []
    for figure in load_figure_registry():
        artifacts.append((figure.svg_path, "figure SVG text nodes"))
        artifacts.append((figure.pdf_path, "standalone figure PDF text"))
    return artifacts


def _iter_pdf_artifacts() -> list[tuple[Path, str]]:
    paths = [
        (PAPER_DIR / "main.pdf", "manuscript PDF text"),
        (PAPER_DIR / "main_review.pdf", "manuscript review PDF text"),
        (PAPER_DIR / "supplementary.pdf", "supplementary PDF text"),
        (PAPER_DIR / "supplementary" / "main_supplement.pdf", "supplementary source PDF text"),
        (PAPER_FIGURES_DIR / "FIGURE_CONTACT_SHEET.pdf", "figure contact sheet PDF text"),
        (PAPER_FIGURES_DIR / "MANUSCRIPT_PAGE_CONTACT_SHEET.pdf", "manuscript contact sheet PDF text"),
        (PAPER_FIGURES_DIR / "EMBEDDED_FIGURE_CONTACT_SHEET.pdf", "embedded figure contact sheet PDF text"),
    ]
    return [(path, source_type) for path, source_type in paths if path.exists()]


def _iter_table_artifacts() -> list[tuple[Path, str]]:
    table_dir = PAPER_DIR / "tables"
    if not table_dir.exists():
        return []
    return [(path, "table LaTeX text") for path in sorted(table_dir.glob("*.tex"))]


def run_visual_text_audit() -> dict[str, Path]:
    findings: list[TextFinding] = []

    for path, source_type in _iter_main_figure_artifacts():
        if not path.exists():
            findings.append(TextFinding(str(path.relative_to(REPO_ROOT)), source_type, "missing artifact", "artifact was not generated"))
            continue
        if path.suffix.lower() == ".svg":
            findings.extend(_scan_text(path, source_type, _extract_svg_text(path)))
        elif path.suffix.lower() == ".pdf":
            findings.extend(_scan_text(path, source_type, _extract_pdf_text(path)))

    for path, source_type in _iter_pdf_artifacts():
        findings.extend(_scan_text(path, source_type, _extract_pdf_text(path)))

    for path, source_type in _iter_table_artifacts():
        findings.extend(_scan_text(path, source_type, _extract_text_file(path)))

    out = REPO_ROOT / "VISUAL_TEXT_AUDIT.md"
    if findings:
        lines = [
            "# Visual Text Audit",
            "",
            "FAILED. The rendered artifact set still contains production-blocking visible text defects.",
            "",
            "| Artifact | Source | Token | Excerpt |",
            "|---|---|---|---|",
        ]
        for finding in findings:
            lines.append(f"| `{finding.artifact}` | {finding.source_type} | `{finding.token}` | {finding.excerpt} |")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        raise RuntimeError(f"Visual text audit failed with {len(findings)} finding(s). See {out}.")

    lines = [
        "# Visual Text Audit",
        "",
        "PASSED.",
        "",
        "Checked SVG text nodes, standalone figure PDFs, final manuscript PDF, supplementary PDF, contact-sheet PDFs and generated LaTeX tables for browser-print artifacts, local paths, raw dataset identifiers, bad capitalization, misspellings and raw snake-case labels.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"visual_text_audit": out}
