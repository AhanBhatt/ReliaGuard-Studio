from __future__ import annotations

import sqlite3
from pathlib import Path

from ..paths import RUNTIME_DIR


STUDY_DB = RUNTIME_DIR / "prospective_study.sqlite"


def connect(path: Path = STUDY_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS participants (
            participant_id TEXT PRIMARY KEY,
            condition TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS responses (
            participant_id TEXT,
            task_id TEXT,
            condition TEXT,
            initial_answer TEXT,
            initial_confidence REAL,
            advice_label TEXT,
            final_answer TEXT,
            final_confidence REAL,
            verification_action TEXT,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return conn

