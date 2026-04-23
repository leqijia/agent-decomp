"""Exp 1 sensitivity: intervene at 25/50/75% of trajectory length regardless of t*.

For each failed trajectory in trajectories/oracle_copies/, compute the target step
at each percentile, pick the closest existing oracle state from oracle/outputs/,
and run a full-oracle intervention there. Writes results to experiments/sensitivity/.

Use closest-available oracle rather than generating fresh ones: Marlin's k=5
batch already covers ~20/40/60/80% of length for each trajectory, so one of those
is always within a few steps of each target percentile.
"""
import json
import os
import re
import sys
sys.path.insert(0, '.')
from harness.evaluator import run_intervention

TRAJ_DIR = 'trajectories/oracle_copies'
ORACLE_DIR = 'oracle/outputs'
RESULTS_DIR = 'experiments/sensitivity'
PERCENTILES = [0.25, 0.50, 0.75]
os.makedirs(RESULTS_DIR, exist_ok=True)


def available_oracle_steps(task_id):
    """Return sorted list of ints: steps where we have an oracle for this task."""
    steps = []
    pat = re.compile(rf'^{task_id}_t(\d+)\.json$')
    for fn in os.listdir(ORACLE_DIR):
        m = pat.match(fn)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def closest(avail, target):
    return min(avail, key=lambda s: abs(s - target)) if avail else None


for fn in sorted(os.listdir(TRAJ_DIR)):
    if not fn.endswith('.json'):
        continue
    task_id = fn.replace('.json', '')
    d = json.load(open(os.path.join(TRAJ_DIR, fn)))
    if d.get('success') is True:
        continue  # only intervene on failures
    n_steps = d.get('total_steps', len(d.get('steps', [])))
    if n_steps < 4:  # too short for meaningful percentiles
        continue
    avail = available_oracle_steps(task_id)
    if not avail:
        print(f'SKIP {task_id}: no oracle outputs')
        continue
    for p in PERCENTILES:
        target = max(1, round(p * n_steps))
        t_star = closest(avail, target)
        out_path = os.path.join(RESULTS_DIR, f'{task_id}_p{int(p*100)}_t{t_star}_result.json')
        if os.path.exists(out_path):
            print(f'SKIP {task_id} p{int(p*100)} already done')
            continue
        oracle_path = os.path.join(ORACLE_DIR, f'{task_id}_t{t_star}.json')
        oracle = json.load(open(oracle_path))
        replacement_context = json.dumps(oracle.get('parsed', {}), indent=2)
        print(f'Running {task_id} p{int(p*100)} target={target} using oracle_t={t_star}')
        result = run_intervention(
            trajectory_path=os.path.join(TRAJ_DIR, fn),
            t_star=t_star,
            replacement_context=replacement_context,
            env_only=False,
            out_path=out_path,
        )
        print(f'  => success={result["success"]} stop={result["stop_reason"]}')

print('Done.')
