from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..paths import RUNTIME_DIR
from ..utils import to_serializable


DB_PATH = RUNTIME_DIR / "research_demo.sqlite"


def get_connection(path: Path | None = None) -> sqlite3.Connection:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path or DB_PATH)


def init_db(path: Path | None = None) -> Path:
    db_path = path or DB_PATH
    with get_connection(db_path) as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                condition_id TEXT,
                task_family TEXT,
                payload_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                report_json TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()
    return db_path


def save_session(session_id: str, user_id: str, condition_id: str, task_family: str, payload: dict[str, Any], path: Path | None = None) -> None:
    init_db(path)
    with get_connection(path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO sessions(session_id, user_id, condition_id, task_family, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_id, condition_id, task_family, json.dumps(to_serializable(payload))),
        )
        connection.commit()


def save_report(session_id: str, report: dict[str, Any], path: Path | None = None) -> None:
    init_db(path)
    with get_connection(path) as connection:
        connection.execute(
            "INSERT INTO reports(session_id, report_json) VALUES (?, ?)",
            (session_id, json.dumps(to_serializable(report))),
        )
        connection.commit()
