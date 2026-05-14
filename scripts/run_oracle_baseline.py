"""Run Exp 1 full-oracle and env-only-control intervention at annotated t*.

By default this runs only the primary resolved t* per trajectory, which is the
headline alpha estimate. Use `--mode all` to run every annotator entry.
"""
import argparse
import json
import os
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass
sys.path.insert(0, '.')
from harness.evaluator import run_intervention

TRAJ_DIRS = ['trajectories/oracle_copies', 'trajectories/data']
RESULTS_DIR = 'experiments/oracle_baseline'
os.makedirs(RESULTS_DIR, exist_ok=True)
RESOLVED_PATH = 'experiments/tstar_resolved.json'


def find_trajectory(task_id):
    for d in TRAJ_DIRS:
        p = os.path.join(d, f'{task_id}.json')
        if os.path.exists(p):
            return p
    return None


def load_all_annotations():
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


def load_primary_annotations():
    """{(annotator, task_id): t_star_step} for the resolved primary t* only."""
    if not os.path.exists(RESOLVED_PATH):
        print(f"WARNING: {RESOLVED_PATH} not found; falling back to all annotations.")
        return load_all_annotations()
    resolved = json.load(open(RESOLVED_PATH))
    out = {}
    for tid, info in resolved.items():
        primary_t = info.get('primary_t_star')
        if primary_t is None or primary_t < 1:
            continue
        for ann in info.get('annotators', []):
            ann_path = f'annotations/{ann}/task_{tid}.json'
            if not os.path.exists(ann_path):
                continue
            d = json.load(open(ann_path))
            if d.get('t_star_step') == primary_t:
                out[(ann, tid)] = primary_t
                break
    return out


def retryable_crash(path):
    """Existing result is not terminal if it crashed before scoring."""
    if not os.path.exists(path):
        return False
    try:
        d = json.load(open(path))
    except json.JSONDecodeError:
        return True
    if d.get('success') is None:
        return True
    if d.get('stop_reason') == 'max_steps' and d.get('intervention') and not d.get('steps'):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--mode',
        choices=['primary', 'all'],
        default='primary',
        help='primary = one resolved t* per task; all = every annotator t* entry.',
    )
    ap.add_argument('--max-steps', type=int, default=50)
    ap.add_argument(
        '--max-new-results',
        type=int,
        default=None,
        help='Stop after writing this many new/rerun result files.',
    )
    ap.add_argument(
        '--max-new-cost-usd',
        type=float,
        default=None,
        help='Stop after newly run result files report this much OpenRouter cost.',
    )
    args = ap.parse_args()

    jobs = load_primary_annotations() if args.mode == 'primary' else load_all_annotations()
    print(f"Found {len(jobs)} annotation entries (mode={args.mode})")
    new_results = 0
    new_cost = 0.0

    def limit_reached():
        if args.max_new_results is not None and new_results >= args.max_new_results:
            return True
        if args.max_new_cost_usd is not None and new_cost >= args.max_new_cost_usd:
            return True
        return False

    for (who, task_id), t_star in sorted(jobs.items()):
        if limit_reached():
            print(
                f"Stopping early: new_results={new_results}, "
                f"new_cost=${new_cost:.4f}"
            )
            break
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

        if os.path.exists(oracle_out) and not retryable_crash(oracle_out):
            print(f'SKIP {task_id} oracle already done')
        else:
            if limit_reached():
                break
            print(f'Running {task_id} t*={t_star} FULL ORACLE')
            result = run_intervention(
                trajectory_path=traj_path,
                t_star=t_star,
                replacement_context=replacement_context,
                max_steps=args.max_steps,
                env_only=False,
                out_path=oracle_out,
            )
            new_results += 1
            new_cost += sum((s.get('cost_usd') or 0) for s in result.get('steps', []))
            print(f'  => success={result["success"]} stop={result["stop_reason"]}')

        if os.path.exists(envonly_out) and not retryable_crash(envonly_out):
            print(f'SKIP {task_id} envonly already done')
        else:
            if limit_reached():
                break
            print(f'Running {task_id} t*={t_star} ENV ONLY')
            result = run_intervention(
                trajectory_path=traj_path,
                t_star=t_star,
                replacement_context='',
                max_steps=args.max_steps,
                env_only=True,
                out_path=envonly_out,
            )
            new_results += 1
            new_cost += sum((s.get('cost_usd') or 0) for s in result.get('steps', []))
            print(f'  => success={result["success"]} stop={result["stop_reason"]}')

    print(f'Done. New/rerun result files: {new_results}; reported new cost: ${new_cost:.4f}')


if __name__ == '__main__':
    main()
