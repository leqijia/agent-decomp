"""
Batch oracle state generation.

Processes trajectory JSON files and generates oracle states at k evenly
spaced steps for each qualifying trajectory (total_steps >= 15).

Usage:
    python oracle/batch_generate.py --input-dir trajectories/data/
    python oracle/batch_generate.py --input-dir trajectories/data/ --stub
    python oracle/batch_generate.py --input-dir trajectories/data/ --limit 3 --stub
"""
import argparse
import json
import os
import sys

# Allow imports from repo root and oracle package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from oracle.generate_oracle import build_prompt, call_oracle, save_output, COST_LOG_PATH

K_STEPS = 5
MIN_STEPS = 15
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def evenly_spaced_steps(total_steps: int, k: int) -> list[int]:
    """Return k evenly spaced step indices from range(1, total_steps+1)."""
    if k == 1:
        return [total_steps]
    step_size = (total_steps - 1) / (k - 1)
    return sorted(set(round(1 + i * step_size) for i in range(k)))


def output_exists(task_id: str, t: int, output_dir: str) -> bool:
    return os.path.exists(os.path.join(output_dir, f"{task_id}_t{t}.json"))


def all_outputs_exist(task_id: str, steps: list[int], output_dir: str) -> bool:
    return all(output_exists(task_id, t, output_dir) for t in steps)


def read_cost_log_total() -> float:
    if not os.path.exists(COST_LOG_PATH):
        return 0.0
    with open(COST_LOG_PATH) as f:
        return json.load(f).get("total_cost_usd", 0.0)


def process_trajectory(traj: dict, use_stub: bool, output_dir: str) -> tuple[int, float]:
    """
    Generate oracle states for one trajectory.

    Returns:
        (states_generated, cost_incurred)
    """
    task_id = traj["task_id"]
    task_goal = traj["intent"]
    total_steps = traj["total_steps"]

    # filter to real step entries only — evaluator error objects lack a 't' field
    steps = [s for s in traj["steps"] if "t" in s]

    target_ts = evenly_spaced_steps(total_steps, K_STEPS)

    # build a lookup by t for quick access
    step_by_t = {s["t"]: s for s in steps}

    states_generated = 0
    cost_incurred = 0.0

    for t in target_ts:
        if output_exists(task_id, t, output_dir):
            continue

        step = step_by_t.get(t)
        if step is None:
            print(f"    WARNING: step t={t} not found in trajectory, skipping")
            continue

        dom = step["observation"]
        prompt = build_prompt(task_goal, steps, dom, t)
        result = call_oracle(prompt, use_stub=use_stub)
        save_output(task_id, t, "v3", result, output_dir=output_dir)

        states_generated += 1
        cost_incurred += result["cost_usd"]

    return states_generated, cost_incurred


def main():
    parser = argparse.ArgumentParser(description="Batch oracle state generation")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing trajectory JSON files")
    parser.add_argument("--stub", action="store_true",
                        help="Use stub (no API calls) for testing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N qualifying trajectories")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory to write oracle outputs (default: oracle/outputs/)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: --input-dir {args.input_dir!r} does not exist.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # collect and filter trajectory files
    all_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".json"))
    qualifying = []
    skipped_schema = 0
    skipped_short = 0

    for fname in all_files:
        fpath = os.path.join(args.input_dir, fname)
        try:
            with open(fpath) as f:
                traj = json.load(f)
            # validate required fields
            for field in ("task_id", "intent", "steps", "total_steps"):
                if field not in traj:
                    raise KeyError(field)
            if traj["total_steps"] < MIN_STEPS:
                skipped_short += 1
                continue
            qualifying.append((fpath, traj))
        except (KeyError, json.JSONDecodeError) as e:
            skipped_schema += 1
            print(f"  SKIP (schema mismatch): {fname} — {e}")

    if args.limit is not None:
        qualifying = qualifying[: args.limit]

    total_trajs = len(qualifying)
    print(f"\nFound {total_trajs} qualifying trajectories "
          f"(skipped {skipped_schema} schema errors, {skipped_short} too short)\n")

    if total_trajs == 0:
        print("Nothing to process.")
        return

    cost_before = read_cost_log_total()
    total_states = 0
    failed = []

    for i, (fpath, traj) in enumerate(qualifying, 1):
        task_id = traj["task_id"]
        total_steps = traj["total_steps"]
        target_ts = evenly_spaced_steps(total_steps, K_STEPS)

        if all_outputs_exist(task_id, target_ts, args.output_dir):
            cost_so_far = read_cost_log_total() - cost_before
            print(f"  [{i}/{total_trajs}] {task_id} steps={total_steps} "
                  f"— all outputs exist, skipping | cost_so_far=${cost_so_far:.4f}")
            continue

        try:
            states, cost = process_trajectory(traj, use_stub=args.stub,
                                              output_dir=args.output_dir)
            total_states += states
            cost_so_far = read_cost_log_total() - cost_before
            print(f"  [{i}/{total_trajs}] {task_id} steps={total_steps} "
                  f"+{states} states | cost_so_far=${cost_so_far:.4f}")
        except Exception as e:
            failed.append((task_id, str(e)))
            print(f"  [{i}/{total_trajs}] {task_id} — ERROR: {e}")

    total_cost = read_cost_log_total() - cost_before

    print(f"\n=== Summary ===")
    print(f"  Trajectories processed: {total_trajs - len(failed)}/{total_trajs}")
    print(f"  Oracle states generated: {total_states}")
    print(f"  Total cost this run:     ${total_cost:.4f}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for task_id, err in failed:
            print(f"    {task_id}: {err}")


if __name__ == "__main__":
    main()
