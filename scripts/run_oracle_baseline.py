import json
import os
import sys
sys.path.insert(0, '.')
from harness.evaluator import run_intervention

TRAJ_DIR = 'trajectories/oracle_copies'
TSTAR_DIR = 'oracle/outputs/tstar/adithya'
RESULTS_DIR = 'experiments/oracle_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)

T_STAR = {
    '114': 17, '322': 2, '332': 2, '401': 4, '415': 9,
    '574': 3, '590': 5, '591': 5, '592': 5, '593': 6,
    '603': 6, '604': 6, '606': 9, '607': 6,
}

for task_id, t_star in sorted(T_STAR.items()):
    oracle_out = os.path.join(RESULTS_DIR, f'{task_id}_tstar_oracle_result.json')
    envonly_out = os.path.join(RESULTS_DIR, f'{task_id}_tstar_envonly_result.json')
    traj_path = os.path.join(TRAJ_DIR, f'{task_id}.json')
    tstar_path = os.path.join(TSTAR_DIR, f'{task_id}_tstar.json')

    if not os.path.exists(traj_path):
        print(f'SKIP {task_id}: trajectory not found')
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
