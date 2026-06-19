from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "paper" / "nmi_submission"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def main() -> int:
    run([sys.executable, "-m", "compileall", "src"])
    run([sys.executable, "-m", "pytest"])
    for command in [
        ["nsca", "screen-datasets"],
        ["nsca", "write-reporting-standard"],
        ["nsca", "run-negative-controls"],
        ["nsca", "run-leakage-audit"],
        ["nsca", "nmi-submission-audit"],
    ]:
        run(command)
    if not (SUBMISSION / "main_nmi_analysis.tex").exists():
        print("No private NMI manuscript found; public-code checks completed.")
        return 0
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(SUBMISSION), str(SUBMISSION / "main_nmi_analysis.tex")])
    run(["bibtex", "main_nmi_analysis"], cwd=SUBMISSION)
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(SUBMISSION), str(SUBMISSION / "main_nmi_analysis.tex")])
    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(SUBMISSION), str(SUBMISSION / "main_nmi_analysis.tex")])
    if (SUBMISSION / "supplementary_information.tex").exists():
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(SUBMISSION), str(SUBMISSION / "supplementary_information.tex")])
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(SUBMISSION), str(SUBMISSION / "supplementary_information.tex")])
    run(["nsca", "nmi-submission-audit"])
    print("NMI submission package rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
