import json
import os
import sys
sys.path.insert(0, '.')
from oracle.generate_oracle import build_prompt, call_oracle

ADITHYA_TSTAR = {
    "114": 17,
    "322": 2,
    "332": 2,
    "401": 4,
    "415": 9,
    "574": 3,
    "590": 5,
    "591": 5,
    "592": 5,
    "593": 6,
    "603": 6,
    "604": 6,
    "606": 9,
    "607": 6,
}

MUHAMMAD_TSTAR = {
    "322": 0,
    "332": 0,
    "401": 3,
    "415": 8,
    "574": 4,
    "603": 0,
    "604": 2,
    "606": 0,
    "607": 0,
}

copies_dir = "trajectories/oracle_copies"
total_cost = 0.0


def generate_tstar_oracles(tstar_dict, annotator, oracle_dir):
    os.makedirs(oracle_dir, exist_ok=True)
    for task_id, t_star in tstar_dict.items():
        if t_star == 0:
            print(f"SKIP {task_id}: t*=0 (annotator marked as unsolvable from start)")
            continue
        outfile = os.path.join(oracle_dir, f"{task_id}_tstar.json")
        if os.path.exists(outfile):
            print(f"SKIP {task_id}: already exists")
            continue
        traj_path = os.path.join(copies_dir, f"{task_id}.json")
        if not os.path.exists(traj_path):
            print(f"SKIP {task_id}: trajectory not found")
            continue
        traj = json.load(open(traj_path))
        steps = [s for s in traj["steps"] if "t" in s]
        steps_up_to_tstar = [s for s in steps if s["t"] <= t_star]
        if not steps_up_to_tstar:
            print(f"SKIP {task_id}: no steps up to t*={t_star}")
            continue
        intent = traj["intent"]
        current_obs = steps_up_to_tstar[-1].get("observation", "")
        prompt = build_prompt(intent, steps_up_to_tstar, current_obs, t_star)
        result = call_oracle(prompt, use_stub=False)
        cost = result.get("cost_usd", 0)
        global total_cost
        total_cost += cost
        output = {
            "task_id": task_id,
            "t_star": t_star,
            "annotator": annotator,
            "failure_classification": f"see annotations/{annotator}",
            "oracle_state": result["content"],
            "cost_usd": cost,
        }
        json.dump(output, open(outfile, "w"), indent=2)
        print(f"Saved {task_id} t*={t_star} cost=${cost:.4f} | total=${total_cost:.4f}")


print("=== Adithya ===")
generate_tstar_oracles(ADITHYA_TSTAR, "Adithya", "oracle/outputs/tstar/adithya")

print("\n=== Muhammad ===")
generate_tstar_oracles(MUHAMMAD_TSTAR, "Muhammad", "oracle/outputs/tstar/muhammad")

print(f"\nDone. Total cost: ${total_cost:.4f}")
