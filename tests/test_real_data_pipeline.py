from __future__ import annotations

from pathlib import Path

from reliaguard_studio.data.adapters.registry import get_dataset_registry
from reliaguard_studio.evaluation.claims_audit import audit_claims
from reliaguard_studio.evaluation.cross_dataset import build_construct_support_matrix
from reliaguard_studio.rules.real_data import build_chi2023_rule_bundle, build_convxai_rule_bundle, build_haiid_rule_bundle


def test_dataset_registry_contains_integrated_and_manual_entries() -> None:
    registry = get_dataset_registry()

    assert "haiid" in registry
    assert "chi2023_dke" in registry
    assert "convxai_iui2025" in registry
    assert registry["haiid"].card.decision == "integrated"
    assert registry["convxai_iui2025"].card.decision == "integrated"
    assert registry["pardos_chatgpt_tutoring"].card.decision == "integrated"
    assert registry["flora_ips"].card.decision == "integrated"


def test_real_rule_bundles_expose_targets_and_rules() -> None:
    haiid_bundle = build_haiid_rule_bundle()
    chi_bundle = build_chi2023_rule_bundle()
    convxai_bundle = build_convxai_rule_bundle()

    assert "overreliance" in haiid_bundle.targets
    assert any(rule.group == "wrong_advice_overreliance" for rule in haiid_bundle.rules)
    assert "appropriate_reliance" in chi_bundle.targets
    assert any(rule.group == "tutorial_protective" for rule in chi_bundle.rules)
    assert "overreliance" in convxai_bundle.targets
    assert any(rule.group == "confidence_inflated_reliance" for rule in convxai_bundle.rules)


def test_construct_matrix_and_claims_audit_smoke() -> None:
    matrix = build_construct_support_matrix()
    assert {"dataset_key", "dataset_name", "appropriate reliance"}.issubset(matrix.columns)

    path = audit_claims(
        {
            "proposed_title": "ReliaGuard Studio: Appropriate Reliance Modeling in Human-AI Decision Making",
            "integrated_datasets": ["haiid", "chi2023_dke", "convxai_iui2025", "pardos_chatgpt_tutoring", "flora_ips"],
        }
    )
    assert isinstance(path, Path)
    assert path.exists()
