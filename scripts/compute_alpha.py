"""Alpha decomposition with primary and upper-bound estimates.

Reads experiments/oracle_baseline/ result files. Each file is named:
    <task_id>_<annotator>_tstar_oracle_result.json
    <task_id>_<annotator>_tstar_envonly_result.json

Two views:
  primary       — uses the LATER of paired t* (proposal §2.2 default).
                  This deflates recovery rate (less time to recover) so
                  alpha_context is a lower bound.
  upper_bound   — uses the EARLIER of paired t* — gives the maximum
                  possible alpha_context.

If even the upper-bound shows alpha_capability dominant, the headline
result is robust to the t*-selection convention.

Output: experiments/alpha_results.json with both views.
"""
import json
import os
import sys
sys.path.insert(0, '.')
from harness.metrics import compute_alpha_decomposition

BASELINE_DIR = 'experiments/oracle_baseline'
RESOLVED_PATH = 'experiments/tstar_resolved.json'
OUT_PATH = 'experiments/alpha_results.json'

# Tasks with broken trajectories or stale annotations — keep narrow.
EXCLUDED_TASKS = set()


def load_results():
    """{(task_id, annotator, condition): result_dict}"""
    out = {}
    for f in sorted(os.listdir(BASELINE_DIR)):
        if not f.endswith('.json'):
            continue
        # Two filename patterns:
        #   <tid>_<annotator>_tstar_<condition>_result.json   (post 2026-04-22)
        #   <tid>_tstar_<condition>_result.json               (older runs, no annotator)
        stem = f.replace('_result.json', '')
        parts = stem.split('_')
        cond = parts[-1]
        if cond not in ('oracle', 'envonly'):
            continue
        if parts[-2] != 'tstar':
            continue
        tid = parts[0]
        annotator = '_'.join(parts[1:-2]) or 'legacy'  # 'annotator_2' or 'legacy'
        try:
            d = json.load(open(os.path.join(BASELINE_DIR, f)))
        except json.JSONDecodeError:
            continue
        out[(tid, annotator, cond)] = d
    return out


def alpha_for(picked_pairs, results):
    """picked_pairs: iterable of (task_id, annotator) — pick those results.
    Returns alpha decomposition or None if too few pairs.
    """
    full = []
    env = []
    for tid, ann in picked_pairs:
        if tid in EXCLUDED_TASKS:
            continue
        r_full = results.get((tid, ann, 'oracle'))
        r_env  = results.get((tid, ann, 'envonly'))
        if r_full is None or r_env is None:
            continue
        if r_full.get('success') is None or r_env.get('success') is None:
            continue
        full.append(r_full); env.append(r_env)
    if len(full) < 3:
        return {'n_pairs': len(full), 'note': 'insufficient data'}
    decomp = compute_alpha_decomposition(full, env)
    decomp['n_pairs'] = len(full)
    return decomp


def main():
    if not os.path.exists(RESOLVED_PATH):
        print(f'Need {RESOLVED_PATH}. Run scripts/build_tstar_resolved.py first.')
        sys.exit(1)
    resolved = json.load(open(RESOLVED_PATH))
    results  = load_results()
    print(f'Loaded {len(results)} oracle/envonly result files for {len(resolved)} resolved tasks.')

    # PRIMARY VIEW: pick the (task, annotator) pair whose t* equals primary_t_star.
    primary_pairs = []
    for tid, info in resolved.items():
        primary_t = info['primary_t_star']
        # find which annotator labelled this t*
        # we have classifications dict whose keys are annotator IDs but not their t*
        # so reload from annotation files
        for ann in info['annotators']:
            ann_path = f'annotations/{ann}/task_{tid}.json'
            if not os.path.exists(ann_path):
                continue
            d = json.load(open(ann_path))
            if d.get('t_star_step') == primary_t:
                primary_pairs.append((tid, ann))
                break

    # UPPER-BOUND VIEW: pick (task, annotator) at earlier t*.
    earlier_pairs = []
    for tid, info in resolved.items():
        earlier_t = info['earlier_t_star']
        if earlier_t is None:
            # singleton — use the only annotator (also goes in earlier set as same value)
            earlier_t = info['primary_t_star']
        for ann in info['annotators']:
            ann_path = f'annotations/{ann}/task_{tid}.json'
            if not os.path.exists(ann_path):
                continue
            d = json.load(open(ann_path))
            if d.get('t_star_step') == earlier_t:
                earlier_pairs.append((tid, ann))
                break

    # ALL-PAIRS VIEW: every (task, annotator) we have results for.
    all_pairs = sorted({(tid, ann) for (tid, ann, _c) in results.keys()})

    primary  = alpha_for(primary_pairs, results)
    upper    = alpha_for(earlier_pairs, results)
    all_view = alpha_for(all_pairs, results)

    out = {
        'primary_later_tstar':   primary,
        'upper_bound_earlier_tstar': upper,
        'all_pairs_pooled':      all_view,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(out, open(OUT_PATH, 'w'), indent=2)

    print('\n=== Primary (later t*, conservative — proposal default) ===')
    _print_decomp(primary)
    print('\n=== Upper bound (earlier t*, max alpha_context) ===')
    _print_decomp(upper)
    print('\n=== All-pairs pooled (sanity) ===')
    _print_decomp(all_view)
    print(f'\nWrote {OUT_PATH}')


def _print_decomp(d):
    if not d or 'alpha_context' not in d:
        print(f'  insufficient data ({d})')
        return
    print(f'  n_pairs={d["n_pairs"]}  R_full={d["R_full"]:.3f}  R_env={d["R_env"]:.3f}')
    print(f'  alpha_context    = {d["alpha_context"]:.3f}  CI {d["ci_95"]["alpha_context"]}')
    print(f'  alpha_env        = {d["alpha_env"]:.3f}  CI {d["ci_95"]["alpha_env"]}')
    print(f'  alpha_capability = {d["alpha_capability"]:.3f}  CI {d["ci_95"]["alpha_capability"]}')


if __name__ == '__main__':
    main()
