import json
import os
import sys
sys.path.insert(0, '.')
from harness.evaluator import run_intervention

ABLATION_DIR = 'oracle/outputs/ablations/tstar'
TRAJ_DIRS = ['trajectories/oracle_copies', 'trajectories/data']
RESULTS_DIR = 'experiments/ablations'
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_trajectory(task_id):
    for d in TRAJ_DIRS:
        p = os.path.join(d, f'{task_id}.json')
        if os.path.exists(p):
            return p
    return None


for fname in sorted(os.listdir(ABLATION_DIR)):
    if not fname.endswith('.json'):
        continue
    out_path = os.path.join(RESULTS_DIR, fname.replace('.json', '_result.json'))
    if os.path.exists(out_path):
        print(f'SKIP {fname} — already done')
        continue
    # Filename shape: "{task_id}_t{t_star}_ablate_{field}.json"
    parts = fname.replace('.json', '').split('_')
    task_id = parts[0]
    field = parts[-1]
    try:
        t_star = int(parts[1].lstrip('t'))
    except (IndexError, ValueError):
        print(f'SKIP {fname} — cannot parse t* from filename')
        continue
    traj_path = find_trajectory(task_id)
    if traj_path is None:
        print(f'SKIP {fname} — trajectory not found in oracle_copies/ or data/')
        continue
    ablated = json.load(open(os.path.join(ABLATION_DIR, fname)))
    replacement_context = json.dumps(ablated.get('parsed', {}), indent=2)
    print(f'Running {fname} t*={t_star} field={field}')
    result = run_intervention(
        trajectory_path=traj_path,
        t_star=t_star,
        replacement_context=replacement_context,
        env_only=False,
        out_path=out_path,
    )
    print(f'  => success={result["success"]} stop={result["stop_reason"]}')

print('Done.')
