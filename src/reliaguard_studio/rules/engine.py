from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd

from ..config.schemas import ProjectConfig, RuleAntecedent, RuleConfig


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + pow(2.718281828, -x))


def _membership(value: float, antecedent: RuleAntecedent) -> float:
    slope = 10.0
    if antecedent.relation == "high":
        return _sigmoid((value - antecedent.threshold) * slope)
    if antecedent.relation == "low":
        return _sigmoid((antecedent.threshold - value) * slope)
    if antecedent.relation == "increasing":
        return _sigmoid((value - antecedent.threshold) * slope)
    if antecedent.relation == "decreasing":
        return _sigmoid((antecedent.threshold - value) * slope)
    if antecedent.relation == "persistent_high":
        return _sigmoid((value - antecedent.threshold) * slope)
    if antecedent.relation == "persistent_low":
        return _sigmoid((antecedent.threshold - value) * slope)
    return 0.0


@dataclass
class RuleActivation:
    rule_id: str
    rule_name: str
    group: str
    activation: float
    confidence: float
    signed_weight: float
    targets: list[str]
    evidence: list[dict[str, float]]
    description: str
    counterfactual_hints: list[str]


class FuzzyTemporalRuleEngine:
    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def evaluate_row(self, row: dict[str, Any] | pd.Series) -> dict[str, Any]:
        sample = dict(row)
        activations: list[RuleActivation] = []
        target_scores: dict[str, float] = {target: 0.0 for target in self.config.targets + self.config.regression_targets}

        for rule in self.config.rules:
            activation = self._evaluate_rule(rule, sample)
            signed_contribution = activation.activation * activation.signed_weight * activation.confidence
            activations.append(activation)
            for target in rule.targets:
                target_scores[target] = target_scores.get(target, 0.0) + signed_contribution

        normalized_scores = {target: max(0.0, min(1.0, 0.5 + value)) for target, value in target_scores.items()}
        top_rules = sorted(activations, key=lambda item: abs(item.activation * item.signed_weight), reverse=True)[:5]
        explanation = {
            "summary": self._build_summary(top_rules, normalized_scores),
            "top_rules": [activation.__dict__ for activation in top_rules],
            "counterfactuals": self._counterfactuals(top_rules),
        }
        return {
            "target_scores": normalized_scores,
            "activations": [activation.__dict__ for activation in activations],
            "explanation": explanation,
        }

    def _evaluate_rule(self, rule: RuleConfig, sample: dict[str, Any]) -> RuleActivation:
        evidence: list[dict[str, float]] = []
        memberships = []
        for antecedent in rule.antecedents:
            value = float(sample.get(antecedent.feature, 0.0))
            membership = _membership(value, antecedent)
            memberships.append(membership)
            evidence.append(
                {
                    "feature": antecedent.feature,
                    "value": value,
                    "threshold": antecedent.threshold,
                    "membership": membership,
                }
            )
        activation_strength = min(memberships) if memberships else 0.0
        return RuleActivation(
            rule_id=rule.id,
            rule_name=rule.name,
            group=rule.group,
            activation=activation_strength,
            confidence=rule.confidence,
            signed_weight=rule.weight,
            targets=rule.targets,
            evidence=evidence,
            description=rule.description,
            counterfactual_hints=rule.counterfactual_hints,
        )

    @staticmethod
    def _build_summary(top_rules: list[RuleActivation], scores: dict[str, float]) -> str:
        if not top_rules:
            return "No symbolic rules were strongly activated."
        highest_target = max(scores, key=scores.get)
        lead = top_rules[0]
        return (
            f"Top symbolic signal: {lead.rule_name} (group={lead.group}, activation={lead.activation:.2f}). "
            f"Highest symbolic target score: {highest_target}={scores[highest_target]:.2f}."
        )

    @staticmethod
    def _counterfactuals(top_rules: list[RuleActivation]) -> list[str]:
        suggestions: list[str] = []
        for rule in top_rules:
            suggestions.extend(rule.counterfactual_hints)
        seen: set[str] = set()
        ordered = []
        for hint in suggestions:
            if hint not in seen:
                seen.add(hint)
                ordered.append(hint)
        return ordered[:5]

    def export_inference_graph(self, evaluation: dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for activation in evaluation["activations"]:
            rule_id = activation["rule_id"]
            graph.add_node(rule_id, kind="rule", group=activation["group"], activation=activation["activation"])
            for evidence in activation["evidence"]:
                feature = evidence["feature"]
                graph.add_node(feature, kind="feature", value=evidence["value"])
                graph.add_edge(feature, rule_id, membership=evidence["membership"])
            for target in activation["targets"]:
                graph.add_node(target, kind="target", score=evaluation["target_scores"].get(target, 0.0))
                graph.add_edge(rule_id, target, weight=activation["signed_weight"])
        return graph
