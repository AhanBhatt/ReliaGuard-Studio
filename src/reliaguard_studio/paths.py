from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASETS_DIR = ARTIFACTS_DIR / "datasets"
EXPERIMENTS_DIR = ARTIFACTS_DIR / "experiments"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
RUNTIME_DIR = ARTIFACTS_DIR / "runtime"
REAL_DATA_DIR = ARTIFACTS_DIR / "real_data"
REAL_DATA_RAW_DIR = REAL_DATA_DIR / "raw"
REAL_DATA_PREPARED_DIR = REAL_DATA_DIR / "prepared"
REAL_DATA_EXPERIMENTS_DIR = REAL_DATA_DIR / "experiments"
REAL_DATA_FIGURES_DIR = REAL_DATA_DIR / "figures"
REAL_DATA_REPORTS_DIR = REAL_DATA_DIR / "reports"
PAPER_DIR = REPO_ROOT / "paper"
PAPER_FIGURES_DIR = PAPER_DIR / "figures"
PAPER_SOURCE_DATA_DIR = PAPER_DIR / "source_data"
PAPER_SUPPLEMENTARY_DIR = PAPER_DIR / "supplementary"
DOCS_DIR = REPO_ROOT / "docs"


def ensure_directories() -> None:
    for directory in [
        ARTIFACTS_DIR,
        DATASETS_DIR,
        EXPERIMENTS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        RUNTIME_DIR,
        REAL_DATA_DIR,
        REAL_DATA_RAW_DIR,
        REAL_DATA_PREPARED_DIR,
        REAL_DATA_EXPERIMENTS_DIR,
        REAL_DATA_FIGURES_DIR,
        REAL_DATA_REPORTS_DIR,
        PAPER_DIR,
        PAPER_FIGURES_DIR,
        PAPER_SOURCE_DATA_DIR,
        PAPER_SUPPLEMENTARY_DIR,
        DOCS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
