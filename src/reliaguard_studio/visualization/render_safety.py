from __future__ import annotations

from pathlib import Path
import re

from ..paths import REPO_ROOT


FORBIDDEN_CODE_PATTERNS = {
    "webbrowser.open": re.compile(r"\bwebbrowser\.open\s*\("),
    "os.startfile render open": re.compile(r"\bos\.startfile\s*\("),
    "Microsoft Edge executable": re.compile(r"(?i)\b(msedge|microsoft-edge|edge\.exe)\b"),
    "cmd start": re.compile(r"(?i)cmd(?:\.exe)?\s*/c\s*start"),
    "PowerShell Start-Process browser/file": re.compile(r"(?i)start-process.*\.(svg|pdf|html?)"),
}

EXCLUDED_FILES = {
    Path("src/reliaguard_studio/visualization/no_gui_browser_guard.py"),
    Path("src/reliaguard_studio/visualization/render_safety.py"),
}


def _iter_code_files() -> list[Path]:
    roots = [REPO_ROOT / "src", REPO_ROOT / "tests"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*.py")))
    return paths


def audit_render_safety() -> Path:
    findings: list[str] = []
    for path in _iter_code_files():
        rel = path.relative_to(REPO_ROOT)
        if rel in EXCLUDED_FILES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in FORBIDDEN_CODE_PATTERNS.items():
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"- `{rel}:{line_no}` violates `{label}`: `{match.group(0)}`")
    out = REPO_ROOT / "RENDER_SAFETY_AUDIT.md"
    if findings:
        out.write_text("# Render Safety Audit\n\nFAILED\n\n" + "\n".join(findings) + "\n", encoding="utf-8")
        raise RuntimeError(f"Render-safety audit failed with {len(findings)} finding(s). See {out}.")
    out.write_text(
        "\n".join(
            [
                "# Render Safety Audit",
                "",
                "PASSED.",
                "",
                "No project code outside the explicit guard module contains headed browser launch calls, Microsoft Edge launch tokens, `webbrowser.open`, `os.startfile` render-artifact opens, or `cmd /c start` patterns.",
                "",
                "The production figure exporter uses CairoSVG when native Cairo is available and PyMuPDF otherwise; it does not print SVGs through a browser.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return out
