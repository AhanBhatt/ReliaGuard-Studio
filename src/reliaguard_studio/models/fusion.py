from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn


def weighted_fusion(neural_score: np.ndarray, symbolic_score: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    return np.clip(alpha * neural_score + (1 - alpha) * symbolic_score, 0.0, 1.0)


def symbolic_posthoc_correction(neural_score: np.ndarray, symbolic_score: np.ndarray, top_rule_activation: np.ndarray) -> np.ndarray:
    correction = 0.12 * symbolic_score + 0.08 * top_rule_activation
    return np.clip(neural_score + correction, 0.0, 1.0)


def uncertainty_aware_fusion(neural_score: np.ndarray, symbolic_score: np.ndarray, neural_uncertainty: np.ndarray) -> np.ndarray:
    certainty = np.clip(1.0 - neural_uncertainty, 0.15, 0.95)
    return np.clip(certainty * neural_score + (1.0 - certainty) * symbolic_score, 0.0, 1.0)


def train_learned_fusion(frame: pd.DataFrame, target: str, feature_names: list[str]) -> LogisticRegression:
    model = LogisticRegression(max_iter=1200, solver="liblinear")
    model.fit(frame[feature_names].to_numpy(dtype=float), frame[target].to_numpy())
    return model


@dataclass
class RuleConstrainedMLPConfig:
    input_dim: int
    hidden_dim: int = 32
    dropout: float = 0.15
    regularization_weight: float = 0.35


class RuleConstrainedMLP(nn.Module):
    def __init__(self, config: RuleConstrainedMLPConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.regularization_weight = config.regularization_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def train_rule_constrained_mlp(
    X: np.ndarray,
    y: np.ndarray,
    symbolic_target: np.ndarray,
    config: RuleConstrainedMLPConfig,
    epochs: int = 25,
    learning_rate: float = 1e-3,
) -> RuleConstrainedMLP:
    model = RuleConstrainedMLP(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    symbolic_t = torch.tensor(symbolic_target, dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_t)
        probs = torch.sigmoid(logits)
        loss = bce(logits, y_t) + config.regularization_weight * torch.mean((probs - symbolic_t) ** 2)
        loss.backward()
        optimizer.step()
    return model


def predict_rule_constrained_mlp(model: RuleConstrainedMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def build_symbolic_feature_frame(rule_evaluations: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for evaluation in rule_evaluations:
        row = {"symbolic_target_score": evaluation["target_scores"].get("overreliance_risk", 0.5)}
        grouped: dict[str, float] = {}
        top_activation = 0.0
        for activation in evaluation["activations"]:
            grouped.setdefault(f"rule_group::{activation['group']}", 0.0)
            grouped[f"rule_group::{activation['group']}"] += activation["activation"] * activation["signed_weight"]
            top_activation = max(top_activation, activation["activation"])
        row.update(grouped)
        row["top_rule_activation"] = top_activation
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)
