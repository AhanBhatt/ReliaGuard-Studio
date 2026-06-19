const datasetNames: Record<string, string> = {
  haiid: "HAIID",
  chi2023_dke: "CHI 2023 DKE",
  convxai_iui2025: "ConvXAI",
  convxai: "ConvXAI",
  pardos_chatgpt_tutoring: "Pardos/Bhandari",
  flora_ips: "FLoRA IPS"
};

const labelOverrides: Record<string, string> = {
  ai: "AI",
  xai: "XAI",
  llm: "LLM",
  llm_agent: "LLM-agent",
  chatgpt: "ChatGPT",
  genai: "GenAI",
  harmful_overreliance: "Harmful overreliance",
  harmful_underreliance: "Harmful underreliance",
  beneficial_ai_reliance: "Beneficial AI reliance",
  correct_self_reliance: "Correct self-reliance",
  independent_correct: "Independent correct",
  independent_incorrect: "Independent incorrect",
  uncertain_disagreement: "Uncertain disagreement",
  inappropriate_reliance: "Inappropriate reliance",
  reliance_state_neurosymbolic: "ReliaGuard-NS",
  uncertainty_aware_fusion: "Uncertainty-aware fusion",
  calibrated_gradient_boosting: "Calibrated gradient boosting",
  learned_fusion: "Learned fusion",
  weighted_fusion: "Weighted fusion",
  logistic_regression: "Logistic regression",
  symbolic_only: "Symbolic only",
  confidence_threshold_gating: "Confidence-threshold gating",
  symbolic_rule_gating: "Symbolic-rule gating",
  conformal_gating: "Conformal gating",
  no_gating: "No gating"
};

export function displayDataset(value: unknown) {
  const key = String(value ?? "").trim();
  return datasetNames[key.toLowerCase()] ?? key;
}

export function displayLabel(value: unknown) {
  const raw = String(value ?? "").trim();
  const key = raw.toLowerCase();
  if (datasetNames[key]) return datasetNames[key];
  if (labelOverrides[key]) return labelOverrides[key];
  return raw
    .replaceAll("_", " ")
    .replaceAll("-", " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => {
      const lower = word.toLowerCase();
      if (labelOverrides[lower]) return labelOverrides[lower];
      if (lower === "ai") return "AI";
      if (lower === "xai") return "XAI";
      if (lower === "llm") return "LLM";
      return lower.charAt(0).toUpperCase() + lower.slice(1);
    })
    .join(" ");
}

export function percent(value: unknown, digits = 0) {
  const numeric = Number(value ?? 0);
  return `${(numeric * 100).toFixed(digits)}%`;
}

export function decimal(value: unknown, digits = 3) {
  return Number(value ?? 0).toFixed(digits);
}

