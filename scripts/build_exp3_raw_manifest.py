"""Build the Experiment 3 raw-condition manifest from existing trajectories.

Thin wrapper around `scripts/_manifest_lib.run_build` with raw-condition
defaults. Tags every trajectory in `--data-dir` as Exp3 raw without modifying
the source files. Emits three artifacts under `--out-dir`:

  manifest.csv  -- full per-task table (one row per scanned trajectory)
  results.json  -- list of result dicts in the shape harness.metrics expects
  skipped.csv   -- anomaly rows (stop_reason in PIPELINE_ANOMALY_REASONS)

Usage:
    python scripts/build_exp3_raw_manifest.py
    python scripts/build_exp3_raw_manifest.py --data-dir trajectories/data \
        --out-dir experiments/exp3/raw --condition raw --experiment-id exp3
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._manifest_lib import run_build


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Experiment 3 raw-condition manifest from existing trajectories."
    )
    parser.add_argument(
        "--data-dir",
        default="trajectories/data",
        help="Directory of trajectory JSON files (default: trajectories/data)",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/exp3/raw",
        help="Output directory (default: experiments/exp3/raw)",
    )
    parser.add_argument(
        "--condition",
        default="raw",
        help="Condition tag stamped into every row (default: raw)",
    )
    parser.add_argument(
        "--experiment-id",
        default="exp3",
        help="Experiment id stamped into every row (default: exp3)",
    )
    args = parser.parse_args()

    return run_build(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        condition=args.condition,
        experiment_id=args.experiment_id,
    )


if __name__ == "__main__":
    sys.exit(main())
