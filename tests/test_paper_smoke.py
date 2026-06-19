from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_paper_build_smoke() -> None:
    if os.environ.get("NSCA_RUN_PRIVATE_PAPER_TESTS") != "1":
        pytest.skip("Private manuscript build is skipped in the public/product test suite.")
    if not Path("paper/main.tex").exists():
        pytest.skip("Private manuscript source is intentionally excluded from public code releases.")
    subprocess.run(["nsca", "build-paper"], check=True)
    pdf = Path("paper/main.pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 0
