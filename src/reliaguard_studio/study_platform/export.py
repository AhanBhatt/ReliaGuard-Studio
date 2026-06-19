from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..paths import REAL_DATA_EXPERIMENTS_DIR
from .storage import STUDY_DB, connect


def export_study_data() -> dict[str, Path]:
    out_dir = REAL_DATA_EXPERIMENTS_DIR / "prospective_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    with connect(STUDY_DB) as conn:
        participants = pd.read_sql_query("SELECT * FROM participants", conn)
        responses = pd.read_sql_query("SELECT * FROM responses", conn)
    participant_path = out_dir / "participants_export.csv"
    response_path = out_dir / "responses_export.csv"
    participants.to_csv(participant_path, index=False)
    responses.to_csv(response_path, index=False)
    return {"participants_export": participant_path, "responses_export": response_path}

