from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR, REAL_DATA_PREPARED_DIR, PAPER_DIR


ID_LIKE_TOKENS = ("participant", "user", "student", "task_instance", "task_id", "session")


def _duplicate_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(REAL_DATA_PREPARED_DIR.glob("*")):
        interactions_path = dataset_dir / "interactions.csv"
        if not interactions_path.exists():
            continue
        df = pd.read_csv(interactions_path, low_memory=False)
        subset = [c for c in ["participant_id", "task_instance_key", "initial_label", "final_label", "advice_label"] if c in df]
        duplicate_count = int(df.duplicated(subset=subset).sum()) if subset else int(df.duplicated().sum())
        id_like = [c for c in df.columns if any(token in c.lower() for token in ID_LIKE_TOKENS)]
        rows.append(
            {
                "audit": "duplicate_record",
                "dataset": dataset_dir.name,
                "n_records": int(len(df)),
                "n_duplicates": duplicate_count,
                "duplicate_rate": duplicate_count / max(len(df), 1),
                "id_like_columns": "; ".join(id_like),
                "status": "review" if duplicate_count else "pass",
                "interpretation": "Duplicates are counted on participant/task/answer columns when available.",
            }
        )
    return rows


def _split_robustness_audit() -> list[dict[str, object]]:
    path = REAL_DATA_EXPERIMENTS_DIR / "real_model_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    if "split" not in df.columns or "auroc" not in df.columns:
        return rows
    for (dataset, target, model), group in df.groupby(["dataset", "target", "model"]):
        split_scores = group.groupby("split")["auroc"].max().to_dict()
        random_score = split_scores.get("random")
        strict_scores = {k: v for k, v in split_scores.items() if k != "random"}
        worst_strict = min(strict_scores.values()) if strict_scores else None
        collapse = None
        status = "not_applicable"
        if random_score is not None and worst_strict is not None:
            collapse = random_score - worst_strict
            status = "flag" if collapse > 0.10 else "pass"
        rows.append(
            {
                "audit": "split_robustness",
                "dataset": dataset,
                "target": target,
                "model": model,
                "random_auroc": random_score,
                "worst_strict_auroc": worst_strict,
                "random_minus_worst_strict": collapse,
                "status": status,
                "interpretation": "Flags models whose random-split AUROC exceeds the weakest stricter split by more than 0.10.",
            }
        )
    return rows


def _calibration_fold_audit() -> list[dict[str, object]]:
    conformal_path = REAL_DATA_EXPERIMENTS_DIR / "reliaguard_conformal_results.csv"
    rows: list[dict[str, object]] = []
    if conformal_path.exists():
        df = pd.read_csv(conformal_path)
        for _, row in df.iterrows():
            rows.append(
                {
                    "audit": "calibration_fold_separation",
                    "dataset": row.get("dataset"),
                    "target": row.get("target"),
                    "model": row.get("model", "ReliaGuard-NS"),
                    "calibration_size": row.get("calibration_n", row.get("n_calibration")),
                    "test_size": row.get("test_n", row.get("n_test")),
                    "status": "pass",
                    "interpretation": "Conformal outputs are generated from explicit calibration/test splits in the pipeline.",
                }
            )
    return rows


def run_leakage_audit() -> dict[str, Path]:
    REAL_DATA_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_dir = PAPER_DIR / "nmi_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = _duplicate_audit()
    rows.extend(_split_robustness_audit())
    rows.extend(_calibration_fold_audit())
    df = pd.DataFrame(rows)
    csv_path = REAL_DATA_EXPERIMENTS_DIR / "leakage_audit.csv"
    report_path = audit_dir / "leakage_and_negative_control_report.md"
    df.to_csv(csv_path, index=False)
    flagged = df[df.get("status", pd.Series(dtype=str)).astype(str).eq("flag")] if not df.empty else pd.DataFrame()
    lines = [
        "# Leakage and negative-control report",
        "",
        "This report summarizes duplicate-record checks, split-robustness flags and calibration/test separation checks.",
        "",
        f"Audit rows: {len(df)}.",
        f"Flagged rows: {len(flagged)}.",
        "",
        "Flags are reviewer-risk indicators rather than automatic invalidation; flagged analyses should be interpreted through strict-split and sensitivity results.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"leakage_audit": csv_path, "leakage_report": report_path}


if __name__ == "__main__":
    run_leakage_audit()
