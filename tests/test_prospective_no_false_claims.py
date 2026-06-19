from __future__ import annotations

from pathlib import Path


def test_prospective_protocol_rejects_false_efficacy_language():
    protocol = Path("paper/nmi_submission/prospective_validation_protocol.md")
    if not protocol.exists():
        return
    text = protocol.read_text(encoding="utf-8").lower()
    forbidden = [
        "completed prospective trial",
        "demonstrated human efficacy",
        "causally reduced harmful reliance",
        "validated deployed intervention",
    ]
    assert not any(term in text for term in forbidden)
    assert "protocol scaffold only" in text
    assert "not be described as human efficacy evidence" in text
