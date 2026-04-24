"""Remove crashed trajectories from batch_log.csv and trajectories/data/.

Usage:
    python scripts/clean_crashed.py            # real run
    python scripts/clean_crashed.py --dry-run  # preview only
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_PATH = ROOT / "trajectories" / "batch_log.csv"
DATA_DIR = ROOT / "trajectories" / "data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print(f"No batch log at {LOG_PATH}")
        return

    with open(LOG_PATH) as f:
        rows = list(csv.reader(f))
    if not rows:
        print("Empty batch log.")
        return

    header, data = rows[0], rows[1:]
    crashed_ids = {
        r[0] for r in data
        if len(r) >= 4 and r[3] == "crash" and r[0].isdigit()
    }
    kept = [r for r in data if not (len(r) >= 4 and r[3] == "crash")]
    existing_jsons = [tid for tid in crashed_ids if (DATA_DIR / f"{tid}.json").exists()]

    print(f"Crashed task IDs in batch_log.csv: {len(crashed_ids)}")
    print(f"Matching JSONs in trajectories/data/: {len(existing_jsons)}")
    print(f"Non-crash rows to keep: {len(kept)} / {len(data)}")

    if args.dry_run:
        print("[dry-run] nothing changed.")
        return

    for tid in existing_jsons:
        (DATA_DIR / f"{tid}.json").unlink()

    with open(LOG_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    remaining = len(list(DATA_DIR.glob("*.json")))
    print(f"Deleted {len(existing_jsons)} JSONs; {remaining} JSONs remain.")
    print(f"Rewrote batch_log.csv with {len(kept)} rows.")


if __name__ == "__main__":
    main()