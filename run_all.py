"""One-shot orchestrator: tune -> experiment -> analysis -> report.

Skips steps whose outputs already exist unless --force is passed. Meant for
producing a fresh report after code changes, without re-running the whole
benchmark from scratch every time.

Run:
    python run_all.py             # skip completed steps
    python run_all.py --force     # re-run everything
    python run_all.py --skip tune # skip tuning (use whatever tuned_params.json exists)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
RESULTS = REPO / "results"

STEPS = [
    ("tune", ["python", "-m", "src.tune"], RESULTS / "tuned_params.json"),
    ("experiment", ["python", "-m", "src.experiment"], RESULTS / "holdout_scores.npz"),
    ("analysis", ["python", "-m", "src.analysis"], RESULTS / "disagreement_rows.csv"),
    ("report", ["python", "-m", "src.report"], RESULTS / "report.md"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-run every step")
    parser.add_argument("--skip", nargs="*", default=[], choices=[s[0] for s in STEPS])
    args = parser.parse_args()

    for name, cmd, marker in STEPS:
        if name in args.skip:
            print(f"[skip] {name}")
            continue
        if marker.exists() and not args.force:
            print(f"[cached] {name} (delete {marker.name} or pass --force to rerun)")
            continue
        print(f"\n[run] {name}: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=REPO)
        if result.returncode != 0:
            print(f"[fail] {name} exited {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll steps done. See results/report.md.")


if __name__ == "__main__":
    main()
