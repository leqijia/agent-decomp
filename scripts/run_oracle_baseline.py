"""Run Exp 1 full-oracle and env-only-control intervention at each annotated t*.

Reads annotations from annotations/annotator_<N>/task_<id>.json and the matching
t*-step oracle from oracle/outputs/tstar/annotator_<N>/<id>_tstar.json. Skips
tasks where either is missing. Writes results to experiments/oracle_baseline/.
"""
import json
import os
import sys
sys.path.insert(0, '.')
from harness.evaluator import run_intervention

TRAJ_DIRS = ['trajectories/oracle_copies', 'trajectories/data']
RESULTS_DIR = 'experiments/oracle_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_trajectory(task_id):
    for d in TRAJ_DIRS:
        p = os.path.join(d, f'{task_id}.json')
        if os.path.exists(p):
            return p
    return None


def load_annotations():
    """{(annotator, task_id): t_star_step} — all 4 annotators."""
    out = {}
    annotator_dirs = [
        ("annotator_1", "annotations/annotator_1"),
        ("annotator_2", "annotations/annotator_2"),
        ("annotator_3", "annotations/annotator_3"),
        ("annotator_4", "annotations/annotator_4"),
    ]
    for who, dir_who in annotator_dirs:
        if not os.path.isdir(dir_who):
            continue
        for fn in os.listdir(dir_who):
            if not fn.startswith("task_") or not fn.endswith(".json"):
                continue
            d = json.load(open(os.path.join(dir_who, fn)))
            tid = fn.replace("task_", "").replace(".json", "")
            t_star = d.get("t_star_step")
            if t_star is not None and t_star >= 1:  # skip t*=0 (unsolvable-from-start)
                out[(who, tid)] = t_star
    return out


jobs = load_annotations()
print(f"Found {len(jobs)} annotation entries")

for (who, task_id), t_star in sorted(jobs.items()):
    tag = f"{task_id}_{who}"
    oracle_out = os.path.join(RESULTS_DIR, f'{tag}_tstar_oracle_result.json')
    envonly_out = os.path.join(RESULTS_DIR, f'{tag}_tstar_envonly_result.json')
    traj_path = find_trajectory(task_id)
    tstar_path = os.path.join(f'oracle/outputs/tstar/{who}', f'{task_id}_tstar.json')

    if traj_path is None:
        print(f'SKIP {task_id}: trajectory not found in oracle_copies/ or data/')
        continue
    if not os.path.exists(tstar_path):
        print(f'SKIP {task_id}: tstar oracle not found')
        continue

    tstar_data = json.load(open(tstar_path))
    parsed = tstar_data.get('parsed', {})
    replacement_context = json.dumps(parsed, indent=2)

    if os.path.exists(oracle_out):
        print(f'SKIP {task_id} oracle already done')
    else:
        print(f'Running {task_id} t*={t_star} FULL ORACLE')
        result = run_intervention(
            trajectory_path=traj_path,
            t_star=t_star,
            replacement_context=replacement_context,
            env_only=False,
            out_path=oracle_out,
        )
        print(f'  => success={result["success"]} stop={result["stop_reason"]}')

    if os.path.exists(envonly_out):
        print(f'SKIP {task_id} envonly already done')
    else:
        print(f'Running {task_id} t*={t_star} ENV ONLY')
        result = run_intervention(
            trajectory_path=traj_path,
            t_star=t_star,
            replacement_context='',
            env_only=True,
            out_path=envonly_out,
        )
        print(f'  => success={result["success"]} stop={result["stop_reason"]}')

print('Done.')
