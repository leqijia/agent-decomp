"""
Oracle state field ablation.

Removes one field at a time from a parsed oracle state and saves the result
as a new file for use in ablation experiments.

Usage:
    python oracle/ablate_oracle.py --input outputs/999_t5.json --field g
    python oracle/ablate_oracle.py --input outputs/999_t5.json --all
"""
import argparse
import copy
import json
import os
import sys

FIELD_EMPTY_VALUES = {
    "g":   "",
    "P_t": [],
    "R_t": [],
    "e_t": "",
    "C":   [],
    "F_t": [],
    "K_t": [],
}

ABLATIONS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "ablations")


def ablate(oracle_data: dict, field: str) -> dict:
    """Return a deep copy of oracle_data with one parsed field replaced by its empty value."""
    ablated = copy.deepcopy(oracle_data)
    if ablated.get("parsed") is None:
        raise ValueError("Oracle file has no 'parsed' field — cannot ablate.")
    ablated["parsed"][field] = FIELD_EMPTY_VALUES[field]
    ablated["ablated_field"] = field
    return ablated


def output_path(task_id, step: int, field: str, out_dir: str) -> str:
    fname = f"{task_id}_t{step}_ablate_{field}.json"
    return os.path.join(out_dir, fname)


def run_ablation(oracle_data: dict, field: str, out_dir: str) -> str:
    task_id = oracle_data["trajectory_id"]
    step = oracle_data["step"]
    ablated = ablate(oracle_data, field)
    path = output_path(task_id, step, field, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ablated, f, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Oracle state field ablation")
    parser.add_argument("--input", required=True,
                        help="Path to oracle output JSON (e.g. outputs/999_t5.json)")
    parser.add_argument("--field", choices=list(FIELD_EMPTY_VALUES),
                        help="Single field to ablate")
    parser.add_argument("--all", action="store_true",
                        help="Ablate all 7 fields")
    parser.add_argument("--output-dir", default=ABLATIONS_DIR,
                        help="Directory to write ablated files")
    args = parser.parse_args()

    if not args.field and not args.all:
        parser.error("Specify --field FIELD or --all")

    if not os.path.isfile(args.input):
        print(f"Error: {args.input!r} not found.")
        sys.exit(1)

    with open(args.input) as f:
        oracle_data = json.load(f)

    fields = list(FIELD_EMPTY_VALUES) if args.all else [args.field]

    for field in fields:
        path = run_ablation(oracle_data, field, args.output_dir)
        print(f"  [{field}] -> {path}")


if __name__ == "__main__":
    main()
