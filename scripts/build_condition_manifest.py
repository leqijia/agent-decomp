"""Build a per-condition manifest for any experimental condition.

Generic counterpart to `scripts/build_exp3_raw_manifest.py`. Expects a
directory of trajectory JSON files (typically `experiments/<exp_id>/<cond>/data/`
or `experiments/baselines/<method>/data/`) and emits the same three sidecar
artifacts (manifest.csv, results.json, skipped.csv) under `--out-dir`, with
the supplied condition / experiment_id stamped into every row.

Usage:
    python scripts/build_condition_manifest.py \\
        --data-dir experiments/exp3/self_generated/data \\
        --out-dir experiments/exp3/self_generated \\
        --condition self_generated --experiment-id exp3

    python scripts/build_condition_manifest.py \\
        --data-dir experiments/baselines/acon/data \\
        --out-dir experiments/baselines/acon \\
        --condition acon --experiment-id baselines

The output schema is defined once in `scripts/_manifest_lib.py` -- every
condition's manifest is column-compatible with every other.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.conditions import VALID_CONDITIONS
from scripts._manifest_lib import run_build


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a per-condition manifest from a directory of trajectory JSONs."
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory of trajectory JSON files for this condition.",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output directory (manifest.csv, results.json, skipped.csv go here).",
    )
    parser.add_argument(
        "--condition", required=True,
        choices=sorted(VALID_CONDITIONS),
        help="Condition tag stamped into every row "
             "(must match harness.conditions.VALID_CONDITIONS).",
    )
    parser.add_argument(
        "--experiment-id", required=True,
        help="Experiment id stamped into every row (e.g. exp3, baselines).",
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
