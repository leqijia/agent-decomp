"""Build task_id -> trajectory_length manifest from a directory of trajectory JSONs.

Usage:
    python scripts/build_length_manifest.py
    python scripts/build_length_manifest.py --in trajectories/data --out experiments/lengths.json
    python scripts/build_length_manifest.py --qualifying-only --min-steps 15
"""
import argparse
import json
import os
import sys

STEP_FIELDS = ["steps", "trajectory", "actions", "history", "events"]
EXPLICIT_LEN_FIELDS = ["trajectory_length", "n_steps", "step_count", "num_steps"]
ID_FIELDS = ["task_id", "trajectory_id", "id"]
BINS = [(0, 10), (11, 20), (21, 40), (41, 80)]


def bin_for_length(L):
    for lo, hi in BINS:
        if lo <= L <= hi:
            return f"{lo}-{hi}"
    return "80+"


def extract_length(data):
    if isinstance(data, list):
        return len(data)
    for field in STEP_FIELDS:
        if field in data and isinstance(data[field], list):
            return len(data[field])
    for field in EXPLICIT_LEN_FIELDS:
        if field in data and isinstance(data[field], int):
            return data[field]
    return None


def extract_task_id(data, stem):
    for field in ID_FIELDS:
        if field in data:
            return str(data[field])
    return stem


def main():
    parser = argparse.ArgumentParser(
        description="Build task_id -> trajectory_length manifest"
    )
    parser.add_argument("--in", dest="in_dir", default="trajectories/data",
                        help="Input directory of trajectory JSON files (default: trajectories/data)")
    parser.add_argument("--out", default="experiments/lengths.json",
                        help="Output JSON path (default: experiments/lengths.json)")
    parser.add_argument("--qualifying-only", action="store_true",
                        help="Exclude trajectories shorter than --min-steps")
    parser.add_argument("--min-steps", type=int, default=15,
                        help="Minimum steps for qualifying filter (default: 15)")
    parser.add_argument("--exclude-stop-reasons", dest="exclude_stop_reasons",
                        nargs="+", default=[],
                        help="Stop reason values to exclude (e.g. crash parse_failures)")
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        print(f"ERROR: input directory not found: {args.in_dir}", file=sys.stderr)
        sys.exit(1)

    manifest = {}
    n_files = 0
    n_skipped = 0
    n_too_short = 0
    n_excluded = 0
    excluded_reasons = {}  # stop_reason -> count

    for fname in sorted(os.listdir(args.in_dir)):
        if not fname.endswith(".json"):
            continue
        n_files += 1
        fpath = os.path.join(args.in_dir, fname)
        stem = os.path.splitext(fname)[0]

        try:
            with open(fpath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: skipping {fname}: {e}")
            n_skipped += 1
            continue

        length = extract_length(data)
        if length is None:
            print(f"  WARNING: could not determine length for {fname}, skipping")
            n_skipped += 1
            continue

        task_id = extract_task_id(data, stem)

        if args.exclude_stop_reasons:
            stop_reason = data.get("stop_reason") if isinstance(data, dict) else None
            if stop_reason is not None and stop_reason in args.exclude_stop_reasons:
                n_excluded += 1
                excluded_reasons[stop_reason] = excluded_reasons.get(stop_reason, 0) + 1
                continue

        if args.qualifying_only and length < args.min_steps:
            n_too_short += 1
            continue

        manifest[task_id] = length

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=2)

    lengths = list(manifest.values())
    print(f"Files scanned:    {n_files}")
    print(f"Manifest entries: {len(manifest)}")
    print(f"Skipped:          {n_skipped}")
    if args.exclude_stop_reasons:
        print(f"Excluded by stop_reason: {n_excluded}")
        if excluded_reasons:
            print("Stop reasons excluded:")
            for reason in sorted(excluded_reasons):
                print(f"  {reason}: {excluded_reasons[reason]}")
    if args.qualifying_only:
        print(f"Too short (<{args.min_steps} steps): {n_too_short}")

    if lengths:
        s = sorted(lengths)
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 == 1 else (s[mid - 1] + s[mid]) / 2
        print(f"Length stats:     min={min(lengths)}  median={median}  max={max(lengths)}")
        bin_counts = {}
        for L in lengths:
            b = bin_for_length(L)
            bin_counts[b] = bin_counts.get(b, 0) + 1
        print("Bin distribution:")
        for lo, hi in BINS:
            b = f"{lo}-{hi}"
            count = bin_counts.get(b, 0)
            pct = 100 * count / len(lengths)
            print(f"  {b:<8} {count:>4}  ({pct:.1f}%)")
        if "80+" in bin_counts:
            count = bin_counts["80+"]
            pct = 100 * count / len(lengths)
            print(f"  {'80+':<8} {count:>4}  ({pct:.1f}%)")
    else:
        print("No entries in manifest.")

    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
