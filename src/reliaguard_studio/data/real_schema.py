from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from ..utils import write_json


class DatasetCard(BaseModel):
    name: str
    short_name: str
    source: str
    source_url: str
    license_name: str
    citation: str
    decision: Literal["integrated", "manual-download only", "rejected", "future work"]
    auto_download: bool
    redistributable: bool | None = None
    participant_or_interaction_count: str
    domains: list[str] = Field(default_factory=list)
    supported_constructs: list[str] = Field(default_factory=list)
    note: str = ""


@dataclass
class PreparedRealDataset:
    metadata: DatasetCard
    interactions: pd.DataFrame
    participants: pd.DataFrame
    tasks: pd.DataFrame
    prepared_dir: Path
    extra_tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def save(self) -> dict[str, Path]:
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "interactions": self.prepared_dir / "interactions.csv",
            "participants": self.prepared_dir / "participants.csv",
            "tasks": self.prepared_dir / "tasks.csv",
            "metadata": self.prepared_dir / "metadata.json",
        }
        self.interactions.to_csv(paths["interactions"], index=False)
        self.participants.to_csv(paths["participants"], index=False)
        self.tasks.to_csv(paths["tasks"], index=False)
        write_json(paths["metadata"], self.metadata.model_dump())
        for name, table in self.extra_tables.items():
            extra_path = self.prepared_dir / f"{name}.csv"
            table.to_csv(extra_path, index=False)
            paths[name] = extra_path
        return paths
