from __future__ import annotations

from reliaguard_studio.config.loader import load_project_config


def test_default_config_loads_and_uses_safe_framing() -> None:
    config = load_project_config()
    assert "Reliance" in config.metadata.manuscript_title
    framing = config.metadata.framing_note.lower()
    assert "public human-ai decision-making datasets" in framing
    assert "not current empirical findings" in framing
    assert len(config.assistance_conditions) == 7
    assert len(config.task_families) == 7
    assert any(metric.id == "verification_robustness" for metric in config.metrics)
