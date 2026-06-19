from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

from .baselines import TARGET_COLUMNS, get_feature_columns


@dataclass
class SequenceDataBundle:
    feature_columns: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_type: Literal["classification", "regression"]


def build_sequence_windows(frame: pd.DataFrame, target: str, sequence_length: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_columns = get_feature_columns(frame, {target})
    sequences = []
    labels = []
    for _, group in frame.sort_values(["user_id", "session_index"]).groupby("user_id"):
        values = group[feature_columns].to_numpy(dtype=np.float32)
        targets = group[target].to_numpy(dtype=np.float32)
        if len(group) < sequence_length:
            continue
        for start in range(0, len(group) - sequence_length + 1):
            end = start + sequence_length
            sequences.append(values[start:end])
            labels.append(targets[end - 1])
    return np.asarray(sequences), np.asarray(labels), feature_columns


def split_sequence_data(frame: pd.DataFrame, target: str, sequence_length: int, seed: int, target_type: Literal["classification", "regression"]) -> SequenceDataBundle:
    X, y, feature_columns = build_sequence_windows(frame, target, sequence_length)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    cutoff = int(len(indices) * 0.75)
    train_idx, test_idx = indices[:cutoff], indices[cutoff:]
    return SequenceDataBundle(
        feature_columns=feature_columns,
        X_train=X[train_idx],
        X_test=X[test_idx],
        y_train=y[train_idx],
        y_test=y[test_idx],
        target_type=target_type,
    )


class RecurrentPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, cell: Literal["gru", "lstm"]) -> None:
        super().__init__()
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        self.recurrent = rnn_cls(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.recurrent(x)
        out = self.dropout(out[:, -1, :])
        return self.output(out).squeeze(-1)


class TransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.proj(x)
        encoded = self.encoder(hidden)
        pooled = self.dropout(encoded[:, -1, :])
        return self.output(pooled).squeeze(-1)


def train_torch_model(
    model: nn.Module,
    data: SequenceDataBundle,
    epochs: int = 18,
    learning_rate: float = 1e-3,
) -> nn.Module:
    X_train = torch.tensor(data.X_train, dtype=torch.float32)
    y_train = torch.tensor(data.y_train, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss() if data.target_type == "classification" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
    return model


def predict_torch_model(model: nn.Module, X: np.ndarray, target_type: Literal["classification", "regression"]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    if target_type == "classification":
        return 1.0 / (1.0 + np.exp(-outputs))
    return outputs


def mc_dropout_uncertainty(model: nn.Module, X: np.ndarray, target_type: Literal["classification", "regression"], passes: int = 20) -> np.ndarray:
    preds = []
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            outputs = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
            if target_type == "classification":
                outputs = 1.0 / (1.0 + np.exp(-outputs))
            preds.append(outputs)
    return np.std(np.vstack(preds), axis=0)
