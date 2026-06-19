from __future__ import annotations

import pandas as pd

from .sensitivity import build_label_definition_sensitivity


def compare_label_definitions(interactions_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Public wrapper for conservative versus alternative reliance labels."""
    frame = build_label_definition_sensitivity(interactions_map)
    if frame.empty:
        return frame
    conservative = frame.loc[frame["definition"].eq("conservative_record_rate"), ["dataset", "outcome", "rate"]].rename(columns={"rate": "conservative_rate"})
    merged = frame.merge(conservative, on=["dataset", "outcome"], how="left")
    merged["delta_from_conservative"] = merged["rate"] - merged["conservative_rate"]
    return merged

