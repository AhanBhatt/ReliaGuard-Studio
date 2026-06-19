from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..paths import PAPER_DIR, PAPER_FIGURES_DIR, REPO_ROOT
from .contact_sheet import ContactFigure, build_contact_sheet
from .figure_registry import load_figure_registry
from .figure_quality import FigureSpec, image_dimensions, score_figure


def _main_figures_from_registry() -> list[FigureSpec]:
    return [
        FigureSpec(
            number=figure.number,
            stem=figure.stem,
            title=figure.title,
            expected_panels=figure.expected_panels,
        )
        for figure in load_figure_registry()
    ]

OBSOLETE_MAIN_FIGURE_STEMS = {
    "figure_01_graphical_abstract",
    "figure_02_evidence_boundary",
    "figure_03_asymmetric_failures",
    "figure_04_interface_calibration",
    "figure_05_learning_process_extension",
    "figure_06_reliance_state_method",
    "figure_07_prediction_generalization",
    "figure_08_calibration_uncertainty",
    "figure_09_rules_counterfactuals",
    "figure_10_policy_frontier",
    "figure_1_graphical_abstract",
    "figure_2_evidence_boundary",
    "figure_3_asymmetric_failures",
    "figure_4_interface_calibration",
    "figure_5_learning_process_extension",
    "figure_6_reliance_state_method",
    "figure_7_prediction_generalization",
    "figure_8_calibration_uncertainty",
    "figure_9_rules_counterfactuals",
    "figure_1_conceptual_framework",
    "figure_2_dataset_map",
    "figure_3_haiid_behavior",
    "figure_4_intervention_calibration",
    "figure_5_reliance_state_model",
    "figure_6_model_comparison",
    "figure_7_calibration",
    "figure_8_rule_analysis",
    "figure_9_policy_evaluation",
    "figure_10_pardos_learning",
}


@dataclass(frozen=True)
class FigureAuditRow:
    figure: str
    path_pdf: str
    path_svg: str
    path_png: str
    referenced_in_latex: bool
    caption_present: bool
    width_px: int
    height_px: int
    vector_outputs: int
    score: float
    issues: str
    action_taken: str


def _read_latex_sources() -> str:
    pieces = []
    for path in [PAPER_DIR / "main.tex", *sorted((PAPER_DIR / "sections").glob("*.tex"))]:
        if path.exists():
            pieces.append(path.read_text(encoding="utf-8"))
    return "\n".join(pieces)


def _caption_present(latex: str, figure_stem: str) -> bool:
    start = latex.find(f"{figure_stem}.pdf")
    return start >= 0 and "\\caption{" in latex[start : start + 1200]


def run_figure_quality_audit() -> dict[str, Path]:
    main_figures = _main_figures_from_registry()
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preview_dir = PAPER_FIGURES_DIR / "preview_png"
    preview_dir.mkdir(parents=True, exist_ok=True)
    numbered_preview_dir = PAPER_FIGURES_DIR / "figure_previews"
    numbered_preview_dir.mkdir(parents=True, exist_ok=True)
    latex = _read_latex_sources()
    audit_existing_text = (REPO_ROOT / "FIGURE_REDESIGN_AUDIT.md").read_text(encoding="utf-8") if (REPO_ROOT / "FIGURE_REDESIGN_AUDIT.md").exists() else ""

    rows: list[FigureAuditRow] = []
    for spec in main_figures:
        stem = spec.stem
        pdf_path = PAPER_FIGURES_DIR / f"{stem}.pdf"
        svg_path = PAPER_FIGURES_DIR / f"{stem}.svg"
        png_path = PAPER_FIGURES_DIR / f"{stem}.png"
        if png_path.exists():
            shutil.copy2(png_path, preview_dir / png_path.name)
            shutil.copy2(png_path, numbered_preview_dir / f"figure_{spec.number:02d}.png")
        width_px, height_px = image_dimensions(png_path)
        referenced = f"{stem}.pdf" in latex
        caption = _caption_present(latex, stem)
        score, issue_list = score_figure(spec, latex, audit_existing_text + "\n" + stem, PAPER_FIGURES_DIR)
        issues = "; ".join(issue_list) if issue_list else "passes automated checks; manual visual inspection still required"
        rows.append(
            FigureAuditRow(
                figure=stem,
                path_pdf=str(pdf_path.relative_to(REPO_ROOT)),
                path_svg=str(svg_path.relative_to(REPO_ROOT)),
                path_png=str(png_path.relative_to(REPO_ROOT)),
                referenced_in_latex=referenced,
                caption_present=caption,
                width_px=width_px,
                height_px=height_px,
                vector_outputs=int(pdf_path.exists()) + int(svg_path.exists()),
                score=round(score, 2),
                issues=issues,
                action_taken="Regenerated under Nature-style visual identity; inspect high-resolution contact sheet before submission.",
            )
        )

    audit_frame = pd.DataFrame([row.__dict__ for row in rows])
    csv_path = PAPER_FIGURES_DIR / "figure_quality_audit.csv"
    audit_frame.to_csv(csv_path, index=False)
    contact_outputs = build_contact_sheet(
        [
            ContactFigure(number=spec.number, stem=spec.stem, title=spec.title, score=float(audit_frame.loc[audit_frame["figure"].eq(spec.stem), "score"].iloc[0]))
            for spec in main_figures
        ],
        preview_dir,
    )

    md_lines = [
        "# Figure Audit",
        "",
    "Automated checks verify file presence, vector output, numbered previews, PNG dimensions, LaTeX references, caption detection, visible-label hygiene, source-data availability and canonical figure numbering. Scores are not a substitute for human visual inspection; the high-resolution contact sheet is generated specifically for that final review.",
        "",
        f"- Contact sheet PDF: `{contact_outputs['figure_contact_sheet_pdf'].relative_to(REPO_ROOT)}`",
        f"- Contact sheet high-resolution PDF: `{contact_outputs['figure_contact_sheet_highres_pdf'].relative_to(REPO_ROOT)}`",
        f"- Contact sheet PNG: `{contact_outputs['figure_contact_sheet_png'].relative_to(REPO_ROOT)}`",
        f"- Numbered previews: `{numbered_preview_dir.relative_to(REPO_ROOT)}`",
        f"- Machine-readable audit: `{csv_path.relative_to(REPO_ROOT)}`",
        "",
        "| Figure | Purpose | Score | Issues | Action taken |",
        "|---|---:|---:|---|---|".replace("---:", "---"),
    ]
    for _, row in audit_frame.iterrows():
        purpose = next(spec.title for spec in main_figures if spec.stem == row["figure"])
        md_lines.append(
            f"| `{row['figure']}` | {purpose} | {row['score']:.0f}/100 | {row['issues']} | {row['action_taken']} |"
        )
    obsolete_present = sorted(stem for stem in OBSOLETE_MAIN_FIGURE_STEMS if any((PAPER_FIGURES_DIR / f"{stem}.{ext}").exists() for ext in ["pdf", "svg", "png"]) or f"{stem}.pdf" in latex)
    md_lines.extend(
        [
            "",
            "## Obsolete Carryover Check",
            "",
            "Obsolete figure stems present: " + (", ".join(f"`{stem}`" for stem in obsolete_present) if obsolete_present else "none"),
            "",
            "## Manual Visual Inspection Checklist",
            "",
            "- Readable at single-column and double-column scale.",
            "- No cropped labels, clipped legends or unreadably rotated axes.",
            "- Every panel has a clear takeaway and panel label.",
            "- No synthetic result is presented as real data.",
            "- Unsupported constructs such as delayed recall and transfer appear only as unsupported future constructs.",
            "- Captions state the empirical conclusion and the evidence boundary.",
        ]
    )
    figure_audit_path = REPO_ROOT / "FIGURE_AUDIT.md"
    figure_audit_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "figure_quality_audit": csv_path,
        **contact_outputs,
        "figure_audit_markdown": figure_audit_path,
    }
