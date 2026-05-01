"""Resolve a single primary t* per trajectory.

Per proposal §2.2:
  - When two annotators disagree, run intervention at both, report the
    later step as the primary result (conservative — less recovery time).
  - Earlier step is reported as the upper bound on alpha_context.

Output:
  experiments/tstar_resolved.json
  {
    "<task_id>": {
      "primary_t_star": int,        # later of the pair, or sole annotator
      "earlier_t_star": int|None,   # earlier of the pair (upper bound), None if singleton
      "annotators": [str, ...],     # who labelled this trajectory
      "classifications": {ann: cls},# their failure_classification calls
      "trajectory_total_steps": int,
      "consensus_class": str|None   # majority class, or None if tied
    }, ...
  }

Singleton-annotated trajectories pass through with primary = annotator's t*.
Trajectories where one annotator marked t*=0 are kept; t*=0 means the agent
was unrecoverable from step 1, so the intervention is at step 1 in practice.
"""
import json
import os
import sys
from collections import Counter

ANNOTATION_DIRS = [
    'annotations/annotator_1',
    'annotations/annotator_2',
    'annotations/annotator_3',
    'annotations/annotator_4',
]
TRAJ_DIRS = ['trajectories/oracle_copies', 'trajectories/data']
OUT_PATH = 'experiments/tstar_resolved.json'


def load_all_annotations():
    """{task_id: [(annotator, t_star, class), ...]}"""
    by_task = {}
    for dir_path in ANNOTATION_DIRS:
        if not os.path.isdir(dir_path):
            continue
        annotator = os.path.basename(dir_path)
        for fn in os.listdir(dir_path):
            if not (fn.startswith('task_') and fn.endswith('.json')):
                continue
            try:
                d = json.load(open(os.path.join(dir_path, fn)))
            except json.JSONDecodeError:
                continue
            tid = str(d['trajectory_id']).replace('task_', '')
            t_star = d.get('t_star_step')
            cls = d.get('failure_classification')
            if t_star is None:
                continue
            by_task.setdefault(tid, []).append((annotator, int(t_star), cls))
    return by_task


def find_trajectory_steps(task_id):
    for d in TRAJ_DIRS:
        p = os.path.join(d, f'{task_id}.json')
        if os.path.exists(p):
            try:
                t = json.load(open(p))
                return t.get('total_steps')
            except json.JSONDecodeError:
                pass
    return None


def majority(values):
    if not values:
        return None
    counts = Counter(v for v in values if v)
    if not counts:
        return None
    top = counts.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None  # tie
    return top[0][0]


def main():
    by_task = load_all_annotations()
    out = {}
    for tid, ann_list in sorted(by_task.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        ann_list.sort(key=lambda a: a[1])  # by t* ascending
        t_stars = [a[1] for a in ann_list]
        annotators = [a[0] for a in ann_list]
        classes = {a[0]: a[2] for a in ann_list}
        if len(ann_list) == 1:
            primary = t_stars[0]
            earlier = None
        else:
            # later = primary, earlier = upper bound
            primary = max(t_stars)
            earlier = min(t_stars)
        out[tid] = {
            'primary_t_star': primary,
            'earlier_t_star': earlier,
            'annotators': annotators,
            'classifications': classes,
            'trajectory_total_steps': find_trajectory_steps(tid),
            'consensus_class': majority(list(classes.values())),
        }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(out, open(OUT_PATH, 'w'), indent=2)
    n_solo   = sum(1 for v in out.values() if v['earlier_t_star'] is None)
    n_paired = sum(1 for v in out.values() if v['earlier_t_star'] is not None)
    n_disagree_class = sum(1 for v in out.values()
                           if v['consensus_class'] is None and len(v['annotators']) > 1)
    print(f'Resolved t* for {len(out)} trajectories.')
    print(f'  Singleton-annotated: {n_solo}')
    print(f'  Paired-annotated:    {n_paired}')
    print(f'  Class disagreement (paired): {n_disagree_class}')
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
