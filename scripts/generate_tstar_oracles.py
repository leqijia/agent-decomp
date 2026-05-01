"""Generate t*-specific oracle states for every annotation on disk.

Reads each annotations/annotator_<N>/task_<id>.json, looks up the trajectory
(oracle_copies/ preferred, falls back to data/), calls the oracle at the
annotated t*, and saves to oracle/outputs/tstar/annotator_<N>/<id>_tstar.json.

Idempotent — skips any (annotator, task) whose output file already exists.
Skips t*=0 annotations (annotator flagged the task as unsolvable from the
start; Exp 1 interventions at t*=0 are handled separately).

Cost: ~$0.05-0.15 per call to Claude Sonnet 4.6 via OpenRouter.
"""
import json
import os
import sys
sys.path.insert(0, '.')
from oracle.generate_oracle import build_prompt, call_oracle

ANNOTATION_DIRS = {
    "annotator_1": "annotations/annotator_1",
    "annotator_2": "annotations/annotator_2",
    "annotator_3": "annotations/annotator_3",
    "annotator_4": "annotations/annotator_4",
}
TRAJ_DIRS = ["trajectories/oracle_copies", "trajectories/data"]
OUT_ROOT = "oracle/outputs/tstar"


def find_trajectory(task_id):
    for d in TRAJ_DIRS:
        p = os.path.join(d, f"{task_id}.json")
        if os.path.exists(p):
            return p
    return None


def generate_for(annotator, ann_dir):
    out_dir = os.path.join(OUT_ROOT, annotator)
    os.makedirs(out_dir, exist_ok=True)
    total = 0.0
    for fn in sorted(os.listdir(ann_dir)):
        if not fn.startswith("task_") or not fn.endswith(".json"):
            continue
        ann = json.load(open(os.path.join(ann_dir, fn)))
        task_id = str(ann.get("trajectory_id", "")).replace("task_", "")
        t_star = ann.get("t_star_step")
        if t_star is None:
            print(f"  skip {task_id}: no t_star_step in annotation")
            continue
        if t_star == 0:
            print(f"  skip {task_id}: t*=0 (task marked unsolvable-from-start)")
            continue
        out_path = os.path.join(out_dir, f"{task_id}_tstar.json")
        if os.path.exists(out_path):
            print(f"  skip {task_id}: already exists at {out_path}")
            continue
        traj_path = find_trajectory(task_id)
        if traj_path is None:
            print(f"  skip {task_id}: trajectory not found in oracle_copies/ or data/")
            continue
        traj = json.load(open(traj_path))
        steps = [s for s in traj["steps"] if "t" in s]
        steps_up_to_tstar = [s for s in steps if s["t"] <= t_star]
        if not steps_up_to_tstar:
            print(f"  skip {task_id}: trajectory has no steps up to t*={t_star}")
            continue
        intent = traj["intent"]
        current_obs = steps_up_to_tstar[-1].get("observation", "")
        prompt = build_prompt(intent, steps_up_to_tstar, current_obs, t_star)
        result = call_oracle(prompt, use_stub=False)
        cost = result.get("cost_usd", 0.0)
        total += cost
        try:
            parsed = json.loads(result["content"])
        except json.JSONDecodeError:
            parsed = None
        output = {
            "trajectory_id": task_id,
            "step": t_star,
            "prompt_version": "v3",
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "cost_usd": cost,
            "raw_response": result["content"],
            "parsed": parsed,
            "annotator": annotator,
            "trajectory_source": traj_path,
        }
        json.dump(output, open(out_path, "w"), indent=2)
        print(f"  saved {task_id} t*={t_star} cost=${cost:.4f}")
    return total


grand_total = 0.0
for annotator, ann_dir in ANNOTATION_DIRS.items():
    if not os.path.isdir(ann_dir):
        continue
    print(f"=== {annotator} ===")
    grand_total += generate_for(annotator, ann_dir)

print(f"\nDone. Total API cost this run: ${grand_total:.4f}")
