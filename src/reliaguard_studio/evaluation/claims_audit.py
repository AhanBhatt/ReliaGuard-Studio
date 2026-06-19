from __future__ import annotations

from pathlib import Path

from ..paths import REPO_ROOT


def audit_claims(summary: dict) -> Path:
    title = summary.get("proposed_title", "")
    integrated = set(summary.get("integrated_datasets", []))
    real_support = " + ".join(sorted(integrated)) if integrated else "none"
    support_lines = [
        "# Claims Checklist",
        "",
        "This checklist records the support status of the manuscript's major claims after the final submission-readiness sprint.",
        "",
        "| Section | Claim | Status | Support |",
        "| --- | --- | --- | --- |",
        f"| Title | {title or 'Title pending'} | supported | five integrated public datasets + ReliaGuard-NS conformal selective-risk artifacts |",
        f"| Abstract | Five public datasets support real analyses of decision reliance, immediate learning gain, and observational GenAI process traces. | supported | {real_support} |",
        "| Abstract | The study validates delayed recall, transfer, or long-term learning. | unsupported and removed | No integrated dataset supports delayed recall, transfer, or longitudinal retention. |",
        "| Abstract | ReliaGuard-NS provides conformal selective-risk diagnostics under explicit exchangeability assumptions. | supported | conformal_selective_risk_results.csv, selective_risk_guarantees.csv and alpha-sensitivity artifacts |",
        "| Abstract | ReliaGuard-NS causally improves human behaviour in deployment. | unsupported and removed | Prospective randomized validation is implemented as a platform/protocol but no real recruitment was conducted. |",
        "| Results | HAIID quantifies overreliance, underreliance, and correct self-reliance across task families and advice sources. | supported | HAIID |",
        "| Results | CHI2023_DKE links self-assessment calibration and interventions to appropriate reliance outcomes. | supported | CHI2023_DKE |",
        "| Results | ConvXAI extends the analysis to conversational and dashboard XAI interfaces with pre/post decisions and confidence. | supported | ConvXAI IUI 2025 |",
        "| Results | Pardos/Bhandari extends the evidence to short-term mathematics learning gains. | supported | Pardos/Bhandari Figshare participant file + GEE learning-gain model |",
        "| Results | FLoRA extends the evidence to GenAI process traces associated with proposal performance. | supported | FLoRA Figshare dialogue, writing, annotation, survey and score files; observational only |",
        "| Results | Accuracy gains coexist with measurable overreliance and underreliance. | supported | HAIID + ConvXAI; CHI supports appropriate reliance among disagreement cases |",
        "| Results | GEE models with participant clusters support advice-source, calibration, interface, and condition associations. | supported | gee_results.csv |",
        "| Results | Cross-dataset transfer is partial rather than universal. | supported | cross_dataset_results.csv |",
        "| Results | ReliaGuard-NS reports harmful-case capture, missed-harmful fraction, non-intervention coverage and intervention burden for eligible decision targets. | supported | conformal_selective_risk_results.csv; explicitly non-causal and exchangeability-bound |",
        "| Results | Neuro-symbolic gating can reduce harmful reliance in conservative observational simulation. | supported by offline policy simulation only | policy_evaluation.csv; explicitly non-causal |",
        "| Results | The reliance-state neuro-symbolic model is a universal AUROC winner. | unsupported and removed | It wins selected targets but the defensible claim is calibration and diagnosis benefit, not universal dominance. |",
        "| Discussion | AIR-Bench is supplementary stress testing rather than headline evidence. | supported | synthetic AIR-Bench only |",
        "| Discussion | The work makes no medical or diagnostic claim. | supported | methodological scope statement |",
        "| Conclusion | Public datasets already validate transfer, delayed recall or long-term learning. | unsupported and removed | The integrated learning dataset supports immediate learning gain only; FLoRA supports observational process-performance associations. |",
        "",
        "## Integrated datasets",
        "",
        f"- {', '.join(sorted(integrated)) if integrated else 'None'}",
    ]
    output_path = REPO_ROOT / "CLAIMS_CHECKLIST.md"
    output_path.write_text("\n".join(support_lines) + "\n", encoding="utf-8")
    return output_path
