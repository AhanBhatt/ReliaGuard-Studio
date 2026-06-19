from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..paths import PAPER_DIR, REPO_ROOT


FORBIDDEN_PLACEHOLDERS = [
    "[?]",
    "??",
    "TODO",
    "TK",
    "pending citation",
    "missing bibliography",
    "private note",
    "pending author review",
    "requires author verification",
    "placeholder",
]

OVERCLAIM_TERMS = [
    "cognitive decline",
    "clinical diagnosis",
    "diagnostic claim",
    "guaranteed acceptance",
    "near-guaranteed",
    "state-of-the-art",
    "universal winner",
    "proven causal",
    "deployment effect",
    "prospective result",
    "human efficacy",
]

ALLOWED_BOUNDARY_PATTERNS = [
    r"does not claim [^.]*causal",
    r"without claiming [^.]*causal",
    r"without claiming [^.]*deployment",
    r"not [^.]*causal",
    r"no [^.]*clinical",
    r"no [^.]*diagnostic",
    r"unsupported",
    r"future",
    r"requires ethics",
    r"requires .*IRB",
]


@dataclass
class AuditResult:
    name: str
    passed: bool
    detail: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _strip_latex_commands(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r" \1 ", text)
    text = text.replace("\\", " ")
    return text


def _abstract_word_count(tex: str) -> int:
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not match:
        return 0
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", _strip_latex_commands(match.group(1))))


def _main_word_count(tex: str) -> int:
    start = tex.find(r"\section{Introduction}")
    end = tex.find(r"\section{Methods}")
    if start == -1 or end == -1 or end <= start:
        return 0
    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", " ", tex[start:end], flags=re.S)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", _strip_latex_commands(body)))


def _display_item_count(tex: str) -> int:
    return len(re.findall(r"\\begin\{figure\}", tex)) + len(re.findall(r"\\begin\{table\}", tex))


def _has_contextual_boundary(text: str, index: int) -> bool:
    window = text[max(0, index - 140) : index + 140].lower()
    return any(re.search(pattern, window) for pattern in ALLOWED_BOUNDARY_PATTERNS)


def run_nmi_submission_audit(run_subprocess_checks: bool = False) -> dict[str, Path]:
    audit_dir = PAPER_DIR / "nmi_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    submission_dir = PAPER_DIR / "nmi_submission"
    tex = submission_dir / "main_nmi_analysis.tex"
    si = submission_dir / "supplementary_information.tex"
    source_dir = submission_dir / "source_data"
    figure_dir = submission_dir / "figures"
    results: list[AuditResult] = []

    manuscript = _read(tex)
    supplement = _read(si)
    combined = "\n".join([manuscript, supplement])

    results.append(AuditResult("manuscript_exists", tex.exists(), str(tex)))
    results.append(AuditResult("supplement_exists", si.exists(), str(si)))

    abstract_words = _abstract_word_count(manuscript)
    results.append(
        AuditResult("abstract_100_150_words", 100 <= abstract_words <= 150, f"abstract_words={abstract_words}")
    )
    main_words = _main_word_count(manuscript)
    results.append(AuditResult("main_text_under_3500_words", main_words <= 3500, f"main_words={main_words}"))
    display_count = _display_item_count(manuscript)
    results.append(AuditResult("display_items_at_most_6", display_count <= 6, f"display_items={display_count}"))

    placeholder_hits = []
    for token in FORBIDDEN_PLACEHOLDERS:
        if token.lower() in combined.lower():
            placeholder_hits.append(token)
    results.append(AuditResult("no_unresolved_placeholders", not placeholder_hits, ", ".join(placeholder_hits)))

    unresolved_cites = re.findall(r"undefined citations|Citation .* undefined|There were undefined references", _read(submission_dir / "main_nmi_analysis.log"))
    results.append(AuditResult("latex_references_resolved", not unresolved_cites, f"hits={len(unresolved_cites)}"))

    source_files = list(source_dir.glob("*.csv")) + list(source_dir.glob("*.json"))
    results.append(AuditResult("source_data_present", len(source_files) >= 6, f"source_data_files={len(source_files)}"))

    figure_missing = []
    for fig in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", manuscript):
        fig_path = figure_dir / fig
        if not fig_path.exists():
            figure_missing.append(fig)
    results.append(AuditResult("all_main_figures_exist", not figure_missing, ", ".join(figure_missing)))

    overclaim_hits = []
    lower = combined.lower()
    for term in OVERCLAIM_TERMS:
        start = 0
        while True:
            idx = lower.find(term.lower(), start)
            if idx == -1:
                break
            if not _has_contextual_boundary(combined, idx):
                overclaim_hits.append(term)
            start = idx + len(term)
    results.append(AuditResult("overclaim_terms_contextualized", not overclaim_hits, ", ".join(sorted(set(overclaim_hits)))))

    if run_subprocess_checks:
        for name, cmd in [
            ("compileall", [sys.executable, "-m", "compileall", "src"]),
            ("pytest", [sys.executable, "-m", "pytest"]),
        ]:
            completed = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            results.append(AuditResult(name, completed.returncode == 0, completed.stdout[-2000:]))

    report = audit_dir / "nmi_submission_audit.md"
    rows = ["# NMI submission audit", ""]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        rows.append(f"- **{result.name}:** {status}. {result.detail}")
    passed = all(r.passed for r in results)
    rows.extend(["", f"Overall: {'PASS' if passed else 'FAIL'}"])
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(f"NMI submission audit failed. See {report}")
    return {"nmi_submission_audit": report}


if __name__ == "__main__":
    run_nmi_submission_audit(run_subprocess_checks=False)
