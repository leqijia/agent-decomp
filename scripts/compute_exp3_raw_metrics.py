"""Compute aggregate metrics for the Experiment 3 raw condition.

Thin wrapper around `scripts.compute_condition_metrics.run_compute` that
defaults to the raw-condition input/output paths so the existing pipeline
keeps working unchanged. The actual computation (per-length-bin success
rates with 95% Wilson CIs, per-site breakdown, stop-reason distribution,
plus avg_tokens / avg_eval_score) is condition-agnostic and lives in the
generic module.

Usage:
    python scripts/compute_exp3_raw_metrics.py
    python scripts/compute_exp3_raw_metrics.py \\
        --results experiments/exp3/raw/results.json \\
        --out experiments/exp3/raw/metrics.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compute_condition_metrics import run_compute


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute Experiment 3 raw-condition metrics from results.json."
    )
    parser.add_argument(
        "--results",
        default="experiments/exp3/raw/results.json",
        help="Path to results.json (default: experiments/exp3/raw/results.json)",
    )
    parser.add_argument(
        "--out",
        default="experiments/exp3/raw/metrics.json",
        help="Output path for metrics.json (default: experiments/exp3/raw/metrics.json)",
    )
    args = parser.parse_args()

    return run_compute(results_path=Path(args.results), out_path=Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
