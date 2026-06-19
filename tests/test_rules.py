from __future__ import annotations

from reliaguard_studio.config.loader import load_project_config
from reliaguard_studio.rules.engine import FuzzyTemporalRuleEngine


def test_rule_engine_flags_overreliance_pattern() -> None:
    config = load_project_config()
    engine = FuzzyTemporalRuleEngine(config)
    sample = {
        "copy_paste_dependence": 0.95,
        "verification_robustness": 0.12,
        "flawed_answer_acceptance": 0.90,
        "confidence": 0.88,
        "immediate_success": 0.92,
        "delayed_recall_score": 0.20,
        "transfer_score": 0.18,
        "citation_support": 0.75,
        "source_checking_rate": 0.05,
        "rolling_offloading": 0.86,
        "offloading_trend": 0.82,
        "calibration_error": 0.74,
        "reflection_depth": 0.18,
    }
    evaluation = engine.evaluate_row(sample)

    assert evaluation["target_scores"]["overreliance_risk"] > 0.7
    top_rule_names = [rule["rule_name"] for rule in evaluation["explanation"]["top_rules"]]
    assert "High copy-paste with low verification" in top_rule_names
    assert evaluation["explanation"]["counterfactuals"]


def test_inference_graph_contains_features_rules_and_targets() -> None:
    config = load_project_config()
    engine = FuzzyTemporalRuleEngine(config)
    evaluation = engine.evaluate_row({"copy_paste_dependence": 0.9, "verification_robustness": 0.1})
    graph = engine.export_inference_graph(evaluation)

    assert "copy_paste_dependence" in graph.nodes
    assert "overreliance_risk" in graph.nodes
    assert any(node.startswith("R00") for node in graph.nodes)
