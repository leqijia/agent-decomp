import json
import os
import sys
sys.path.insert(0, '.')
from harness.evaluator import run_intervention

ABLATION_DIR = 'oracle/outputs/ablations/tstar'
TRAJ_DIR = 'trajectories/oracle_copies'
RESULTS_DIR = 'experiments/ablations'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Adithya t* values
T_STAR = {
    '114': 17, '322': 2, '332': 2, '401': 4, '415': 9,
    '574': 3, '590': 5, '591': 5, '592': 5, '593': 6,
    '603': 6, '604': 6, '606': 9, '607': 6,
}

for fname in sorted(os.listdir(ABLATION_DIR)):
    if not fname.endswith('.json'):
        continue
    out_path = os.path.join(RESULTS_DIR, fname.replace('.json', '_result.json'))
    if os.path.exists(out_path):
        print(f'SKIP {fname} — already done')
        continue
    parts = fname.replace('.json', '').split('_')
    task_id = parts[0]
    field = parts[-1]
    t_star = T_STAR.get(task_id)
    if t_star is None:
        print(f'SKIP {fname} — no t* found')
        continue
    traj_path = os.path.join(TRAJ_DIR, f'{task_id}.json')
    if not os.path.exists(traj_path):
        print(f'SKIP {fname} — trajectory not found')
        continue
    ablated = json.load(open(os.path.join(ABLATION_DIR, fname)))
    replacement_context = json.dumps(ablated.get('parsed', {}), indent=2)
    print(f'Running {fname} t*={t_star} field={field}')
    result = run_intervention(
        trajectory_path=traj_path,
        t_star=t_star,
        replacement_context=replacement_context,
        out_path=out_path,
    )
    print(f'  => success={result["success"]} stop={result["stop_reason"]}')

print('Done.')
