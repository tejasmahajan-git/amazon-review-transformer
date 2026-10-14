
from pathlib import Path
import subprocess
import sys

import pandas as pd

from audit import run_audit
from data_prep import run_data_prep


ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
DATA = ROOT / "data" / "amazon_electronics_combined.csv"


def run_script(script_name: str, phase_name: str) -> None:
    """Run a project script and stop if it fails."""
    script_path = SRC / script_name

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Cannot run {phase_name}: script not found at {script_path}"
        )

    print(f"\n{'=' * 60}")
    print(f"RUNNING: {phase_name}")
    print(f"{'=' * 60}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=ROOT,
        check=True,
    )

    print(f"COMPLETED: {phase_name}")


def main():
    print("=" * 60)
    print("AMAZON CUSTOMER INTELLIGENCE PIPELINE")
    print("=" * 60)

    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA}")

    # Phase 1: Audit
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(DATA, low_memory=False)
    if df.empty:
        raise ValueError("The dataset contains no rows.")
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    print("\n[2/6] Data audit...")
    run_audit(df)

    # Phase 2: Data preparation
    print("\n[3/6] Data preparation...")
    run_data_prep(df)

    # Phase 3: Baseline sentiment analysis
    run_script(
        "sentiment_analysis.py",
        "Baseline sentiment analysis",
    )

    # Phase 4: DistilBERT sentiment analysis
    run_script(
        "sentiment_distilbert.py",
        "DistilBERT sentiment analysis",
    )

    # Phase 5: Dashboard
    print("\n[6/6] Launching dashboard...")
    dashboard_path = SRC / "app.py"

    if not dashboard_path.is_file():
        raise FileNotFoundError(
            f"Dashboard not found at {dashboard_path}"
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            f"\nPIPELINE STOPPED: a phase failed "
            f"with exit code {exc.returncode}."
        )
        raise
