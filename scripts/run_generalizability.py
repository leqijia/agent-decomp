"""Exp 1 generalizability run with GPT-5.2 as the task agent.

Per proposal §3 (Model generalizability): run Experiment 1 (oracle full +
env-only) on a 100-task subset using GPT-5.2 as the task agent. The
oracle generator (Claude Sonnet 4.6) and the t* labels stay the same —
only the agent that resumes after t* changes.

This script reuses the existing tstar oracle outputs and annotations.
For each (task, annotator) pair where we have a t* oracle, it runs
two interventions with the GPT-5.2 model:
  - full oracle (replacement_context = serialized oracle state)
  - env_only   (replacement_context = '', env_only=True)

Output: experiments/oracle_baseline_gpt52/<task>_<annotator>_tstar_<cond>_result.json

Subset selection:
  --subset-size N  Pick N tasks stratified across length bins (default 100).
                   Pulled from experiments/tstar_resolved.json (the
                   annotated set). If N exceeds available, uses all.

Usage on the VM:
  python scripts/run_generalizability.py --subset-size 100 \
      --model openai/gpt-5.2 --max-cost-usd 400
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, '.')

from harness.evaluator import run_intervention
from llm.config import GENERALIZABILITY_MODEL


RESOLVED_PATH = 'experiments/tstar_resolved.json'
TSTAR_DIR = 'oracle/outputs/tstar'
TRAJ_DIRS = ['trajectories/oracle_copies', 'trajectories/data']
OUT_DIR = 'experiments/oracle_baseline_gpt52'


def find_trajectory(task_id):
    for d in TRAJ_DIRS:
        p = os.path.join(d, f'{task_id}.json')
        if os.path.exists(p):
            return p
    return None


def stratified_pick(resolved, size):
    """Pick `size` tasks stratified by trajectory length bin."""
    bins = {'<=10': [], '11-20': [], '21-40': [], '41+': []}
    for tid, info in resolved.items():
        L = info.get('trajectory_total_steps') or 0
        if L <= 10: bins['<=10'].append(tid)
        elif L <= 20: bins['11-20'].append(tid)
        elif L <= 40: bins['21-40'].append(tid)
        else: bins['41+'].append(tid)
    per_bin = max(1, size // 4)
    picked = []
    for k in ['<=10', '11-20', '21-40', '41+']:
        picked.extend(sorted(bins[k], key=int)[:per_bin])
    # if we under-picked, top up from the largest bin
    rest = [t for t in resolved if t not in picked]
    while len(picked) < size and rest:
        picked.append(rest.pop(0))
    return picked[:size]


def find_tstar_oracle(task_id, annotator):
    p = os.path.join(TSTAR_DIR, annotator, f'{task_id}_tstar.json')
    return p if os.path.exists(p) else None


def retryable_crash(path):
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
    except json.JSONDecodeError:
        return True
    return d.get('success') is None and d.get('stop_reason') == 'crash'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset-size', type=int, default=100)
    ap.add_argument('--model', default=GENERALIZABILITY_MODEL)
    ap.add_argument('--max-cost-usd', type=float, default=400.0,
                    help='Stop early if running spend exceeds this estimate.')
    ap.add_argument('--out-dir', default=OUT_DIR)
    ap.add_argument('--max-steps', type=int, default=75)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    resolved = json.load(open(RESOLVED_PATH))
    picked = stratified_pick(resolved, args.subset_size)
    print(f'Picked {len(picked)} tasks for generalizability run with {args.model}')
    print(f'Output dir: {args.out_dir}')
    print(f'Cost cap: ${args.max_cost_usd:.2f}')

    spend = 0.0
    n_done = 0
    n_skip = 0

    for tid in picked:
        info = resolved[tid]
        primary_t = info['primary_t_star']
        # Pick the annotator whose t* matches primary
        chosen_ann = None
        for ann in info['annotators']:
            ann_path = f'annotations/{ann}/task_{tid}.json'
            if not os.path.exists(ann_path):
                continue
            d = json.load(open(ann_path))
            if d.get('t_star_step') == primary_t:
                chosen_ann = ann
                break
        if chosen_ann is None:
            print(f'  [{tid}] no annotator at primary t*={primary_t}, skipping')
            n_skip += 1
            continue

        oracle_tstar_path = find_tstar_oracle(tid, chosen_ann)
        if oracle_tstar_path is None:
            print(f'  [{tid}] no t*-oracle at {chosen_ann}, skipping (run generate_tstar_oracles.py first)')
            n_skip += 1
            continue
        traj_path = find_trajectory(tid)
        if traj_path is None:
            print(f'  [{tid}] trajectory missing, skipping')
            n_skip += 1
            continue

        oracle_state = json.load(open(oracle_tstar_path))
        replacement_context = json.dumps(oracle_state.get('parsed', {}), indent=2)

        tag = f'{tid}_{chosen_ann}'
        out_full = os.path.join(args.out_dir, f'{tag}_tstar_oracle_result.json')
        out_env = os.path.join(args.out_dir, f'{tag}_tstar_envonly_result.json')

        if not os.path.exists(out_full) or retryable_crash(out_full):
            print(f'  [{tid}] FULL  t*={primary_t} ', end='', flush=True)
            t0 = time.time()
            try:
                r = run_intervention(
                    trajectory_path=traj_path,
                    t_star=primary_t,
                    replacement_context=replacement_context,
                    model=args.model,
                    max_steps=args.max_steps,
                    env_only=False,
                    out_path=out_full,
                )
                ep_cost = sum((s.get('cost_usd') or 0) for s in r.get('steps', []))
                spend += ep_cost
                print(f'success={r.get("success")} stop={r.get("stop_reason")} cost=${ep_cost:.3f} dt={int(time.time()-t0)}s spent=${spend:.2f}')
            except Exception as e:
                print(f'ERROR: {e}')
        else:
            print(f'  [{tid}] FULL already exists — skip')

        if not os.path.exists(out_env) or retryable_crash(out_env):
            print(f'  [{tid}] ENV   t*={primary_t} ', end='', flush=True)
            t0 = time.time()
            try:
                r = run_intervention(
                    trajectory_path=traj_path,
                    t_star=primary_t,
                    replacement_context='',
                    model=args.model,
                    max_steps=args.max_steps,
                    env_only=True,
                    out_path=out_env,
                )
                ep_cost = sum((s.get('cost_usd') or 0) for s in r.get('steps', []))
                spend += ep_cost
                print(f'success={r.get("success")} stop={r.get("stop_reason")} cost=${ep_cost:.3f} dt={int(time.time()-t0)}s spent=${spend:.2f}')
            except Exception as e:
                print(f'ERROR: {e}')
        else:
            print(f'  [{tid}] ENV already exists — skip')

        n_done += 1
        if spend > args.max_cost_usd:
            print(f'\nSPENT ${spend:.2f} > cap ${args.max_cost_usd:.2f}; stopping.')
            break

    print(f'\nDone. {n_done} tasks completed, {n_skip} skipped. Total spend ~${spend:.2f}.')


if __name__ == '__main__':
    main()
