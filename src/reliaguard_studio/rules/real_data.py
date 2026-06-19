from __future__ import annotations

from dataclasses import dataclass

from ..config.schemas import RuleAntecedent, RuleConfig


@dataclass(frozen=True)
class RealRuleBundle:
    dataset_name: str
    targets: list[str]
    regression_targets: list[str]
    rules: list[RuleConfig]


def _rule(
    rule_id: str,
    name: str,
    group: str,
    description: str,
    weight: float,
    confidence: float,
    targets: list[str],
    antecedents: list[tuple[str, str, float]],
    counterfactual_hints: list[str],
) -> RuleConfig:
    return RuleConfig(
        id=rule_id,
        name=name,
        group=group,
        description=description,
        weight=weight,
        confidence=confidence,
        temporal_window=1,
        targets=targets,
        antecedents=[RuleAntecedent(feature=feature, relation=relation, threshold=threshold) for feature, relation, threshold in antecedents],
        counterfactual_hints=counterfactual_hints,
    )


def build_haiid_rule_bundle() -> RealRuleBundle:
    rules = [
        _rule(
            "haiid_over_wrong_advice",
            "Wrong-advice overreliance vulnerability",
            "wrong_advice_overreliance",
            "Low-confidence participants facing wrong advice from a high-stated-accuracy source are vulnerable to harmful reliance.",
            weight=0.36,
            confidence=0.86,
            targets=["overreliance", "harmful_reliance"],
            antecedents=[
                ("initial_correct", "high", 0.50),
                ("advice_wrong", "high", 0.50),
                ("initial_confidence", "low", 0.45),
                ("stated_accuracy_normalized", "high", 0.75),
            ],
            counterfactual_hints=[
                "Reduce stated certainty or ask for independent justification before revealing advice.",
                "Flag disagreements when the user was initially correct but uncertain.",
            ],
        ),
        _rule(
            "haiid_under_correct_advice",
            "Correct-advice underreliance vulnerability",
            "correct_advice_underreliance",
            "Confident initial errors can trigger rejection of correct advice.",
            weight=0.34,
            confidence=0.83,
            targets=["underreliance", "harmful_reliance"],
            antecedents=[
                ("initial_incorrect", "high", 0.50),
                ("advice_correct", "high", 0.50),
                ("initial_confidence", "high", 0.60),
            ],
            counterfactual_hints=[
                "Ask the participant to compare evidence before dismissing correct advice.",
                "Encourage explicit error-checking when confidence is high but accuracy is poor.",
            ],
        ),
        _rule(
            "haiid_beneficial_low_conf",
            "Low-confidence beneficial reliance",
            "low_confidence_beneficial_reliance",
            "Correct advice can be beneficial when users begin with low-confidence errors.",
            weight=0.30,
            confidence=0.79,
            targets=["appropriate_reliance"],
            antecedents=[
                ("initial_incorrect", "high", 0.50),
                ("advice_correct", "high", 0.50),
                ("initial_confidence", "low", 0.40),
            ],
            counterfactual_hints=[
                "Preserve support for low-confidence correction while keeping verification visible.",
            ],
        ),
        _rule(
            "haiid_correct_self_reliance",
            "Correct self-reliance under disagreement",
            "correct_self_reliance",
            "High-confidence correct users can appropriately reject wrong advice.",
            weight=0.30,
            confidence=0.81,
            targets=["appropriate_reliance"],
            antecedents=[
                ("initial_correct", "high", 0.50),
                ("advice_wrong", "high", 0.50),
                ("initial_confidence", "high", 0.60),
            ],
            counterfactual_hints=[
                "Preserve user agency when their own signal is strong and the advice conflicts.",
            ],
        ),
        _rule(
            "haiid_ai_source_susceptibility",
            "Advice-source susceptibility",
            "advice_source_susceptibility",
            "Participants are more at risk when AI-labelled advice is paired with high stated accuracy.",
            weight=0.18,
            confidence=0.74,
            targets=["overreliance"],
            antecedents=[
                ("advice_source_ai", "high", 0.50),
                ("stated_accuracy_normalized", "high", 0.75),
            ],
            counterfactual_hints=[
                "Make uncertainty visible rather than relying on source prestige.",
            ],
        ),
        _rule(
            "haiid_task_susceptibility",
            "Task-domain susceptibility",
            "task_domain_susceptibility",
            "Some task families exhibit systematically higher overreliance rates.",
            weight=0.12,
            confidence=0.70,
            targets=["overreliance"],
            antecedents=[
                ("task_family_overreliance_rate", "high", 0.12),
            ],
            counterfactual_hints=[
                "Prioritize safeguards in domains with historically elevated overreliance.",
            ],
        ),
    ]
    return RealRuleBundle(dataset_name="haiid", targets=["overreliance", "underreliance", "appropriate_reliance", "harmful_reliance"], regression_targets=[], rules=rules)


def build_chi2023_rule_bundle() -> RealRuleBundle:
    rules = [
        _rule(
            "chi_overconfidence_underreliance",
            "Overconfidence-linked underreliance",
            "correct_advice_underreliance",
            "Participants who overestimated their first-batch performance are at risk of rejecting correct advice later.",
            weight=0.34,
            confidence=0.84,
            targets=["underreliance", "harmful_reliance"],
            antecedents=[
                ("first_batch_overestimation", "high", 0.50),
                ("advice_correct", "high", 0.50),
                ("initial_incorrect", "high", 0.50),
                ("tutorial_present", "low", 0.50),
            ],
            counterfactual_hints=[
                "Add calibration feedback before a second round of AI-assisted decisions.",
                "Use short tutorials to reveal AI fallibility and user error patterns.",
            ],
        ),
        _rule(
            "chi_tutorial_protective",
            "Tutorial protective rule",
            "tutorial_protective",
            "Tutorials can improve appropriate reliance among disagreement cases.",
            weight=0.28,
            confidence=0.80,
            targets=["appropriate_reliance"],
            antecedents=[
                ("tutorial_present", "high", 0.50),
                ("advice_correct", "high", 0.50),
                ("initial_incorrect", "high", 0.50),
            ],
            counterfactual_hints=[
                "Retain tutorial-style fallibility cues before assisted decisions.",
            ],
        ),
        _rule(
            "chi_xai_overreliance",
            "XAI-amplified overreliance",
            "wrong_advice_overreliance",
            "Explanation-rich interfaces can still amplify reliance when the advice is wrong.",
            weight=0.22,
            confidence=0.76,
            targets=["overreliance", "harmful_reliance"],
            antecedents=[
                ("xai_present", "high", 0.50),
                ("advice_wrong", "high", 0.50),
                ("initial_correct", "high", 0.50),
            ],
            counterfactual_hints=[
                "Pair explanations with prompts that require checking disagreement rather than simply reading the rationale.",
            ],
        ),
        _rule(
            "chi_high_trust_wrong_advice",
            "High-trust wrong-advice vulnerability",
            "confident_wrong_advice_vulnerability",
            "High trust combined with wrong advice can increase harmful agreement.",
            weight=0.18,
            confidence=0.72,
            targets=["overreliance"],
            antecedents=[
                ("batch_trust", "high", 0.65),
                ("advice_wrong", "high", 0.50),
                ("initial_correct", "high", 0.50),
            ],
            counterfactual_hints=[
                "Surface uncertainty and encourage explanation checking when trust is high.",
            ],
        ),
    ]
    return RealRuleBundle(dataset_name="chi2023_dke", targets=["overreliance", "underreliance", "appropriate_reliance", "harmful_reliance"], regression_targets=[], rules=rules)


def build_convxai_rule_bundle() -> RealRuleBundle:
    rules = [
        _rule(
            "convxai_llm_wrong_advice",
            "LLM-agent wrong-advice overreliance",
            "wrong_advice_overreliance",
            "Conversational LLM-agent explanations can increase risk when a user starts correct and the model advice is wrong.",
            weight=0.30,
            confidence=0.80,
            targets=["overreliance", "harmful_reliance"],
            antecedents=[
                ("llm_agent", "high", 0.50),
                ("advice_wrong", "high", 0.50),
                ("initial_correct", "high", 0.50),
            ],
            counterfactual_hints=[
                "Add a disagreement check before showing fluent explanations for wrong model predictions.",
                "Require the user to state evidence against the model before adopting it.",
            ],
        ),
        _rule(
            "convxai_high_reliability_wrong_advice",
            "High perceived reliability under wrong advice",
            "confidence_inflated_reliance",
            "High post-explanation reliability ratings can be dangerous when the model prediction is wrong.",
            weight=0.24,
            confidence=0.74,
            targets=["overreliance", "harmful_reliance"],
            antecedents=[
                ("post_explain_reliability", "high", 0.70),
                ("advice_wrong", "high", 0.50),
                ("initial_correct", "high", 0.50),
            ],
            counterfactual_hints=[
                "Surface model fallibility and ask for a second independent check when reliability ratings are high.",
            ],
        ),
        _rule(
            "convxai_beneficial_low_confidence",
            "Low-confidence beneficial reliance",
            "low_confidence_beneficial_reliance",
            "Correct model predictions can help when users begin with low-confidence incorrect judgments.",
            weight=0.27,
            confidence=0.78,
            targets=["appropriate_reliance"],
            antecedents=[
                ("initial_incorrect", "high", 0.50),
                ("advice_correct", "high", 0.50),
                ("initial_confidence", "low", 0.55),
            ],
            counterfactual_hints=[
                "Preserve explanatory support for low-confidence errors while keeping uncertainty explicit.",
            ],
        ),
        _rule(
            "convxai_underreliance_high_confidence",
            "High-confidence underreliance",
            "correct_advice_underreliance",
            "High initial confidence can make users ignore correct model predictions.",
            weight=0.26,
            confidence=0.77,
            targets=["underreliance", "harmful_reliance"],
            antecedents=[
                ("initial_incorrect", "high", 0.50),
                ("advice_correct", "high", 0.50),
                ("initial_confidence", "high", 0.70),
            ],
            counterfactual_hints=[
                "Ask users to explicitly compare their rationale with the model evidence before rejecting correct advice.",
            ],
        ),
        _rule(
            "convxai_engagement_protective",
            "Interaction engagement protective rule",
            "process_engagement_protective",
            "Richer interaction with explanation tools can support appropriate reliance when advice is correct.",
            weight=0.18,
            confidence=0.69,
            targets=["appropriate_reliance"],
            antecedents=[
                ("user_question_rate", "high", 0.30),
                ("advice_correct", "high", 0.50),
                ("initial_incorrect", "high", 0.50),
            ],
            counterfactual_hints=[
                "Encourage question asking and evidence inspection before final commitment.",
            ],
        ),
    ]
    return RealRuleBundle(dataset_name="convxai_iui2025", targets=["overreliance", "underreliance", "appropriate_reliance", "harmful_reliance"], regression_targets=[], rules=rules)
