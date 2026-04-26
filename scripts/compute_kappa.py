"""Three-tier inter-annotator agreement.

Tier 1 — Fleiss's kappa across all 4 annotators on the shared 20 trajectories.
Tier 2 — Cohen's kappa for Adithya (ann1) vs Muhammad (ann2) on their 40.
Tier 3 — Cohen's kappa for Marlin  (ann3) vs Rocky   (ann4) on their 40.
"""
import json
import os
import sys
sys.path.insert(0, '.')
from harness.metrics import compute_cohens_kappa

# ---------------------------------------------------------------------------
# Trajectory ID sets — fill in actual IDs as annotation batches land
# ---------------------------------------------------------------------------

# All 4 annotators annotated these — Fleiss's kappa only
SHARED_20 = [28, 49, 102, 103, 104, 159, 171, 178, 181, 241,
             12, 13, 27, 62, 114, 142, 162, 169, 239, 273]

# Adithya + Muhammad only — Cohen's kappa only (do NOT include SHARED_20)
ADITHYA_MUHAMMAD_40 = [270, 277, 285, 309, 312, 351, 352, 375, 390, 393,
                       399, 411, 412, 432, 434, 442, 446, 451, 454, 460,
                       471, 473, 484, 486, 489, 494, 496, 502, 522, 528,
                       575, 594, 615, 653, 654, 655, 672, 675, 682, 688]

# Marlin + Rocky only — Cohen's kappa only (do NOT include SHARED_20)
MARLIN_ROCKY_40 = [280, 311, 317, 324, 328, 354, 388, 391, 400, 413,
                   420, 422, 433, 437, 441, 443, 444, 447, 458, 461,
                   472, 481, 482, 483, 487, 488, 492, 507, 508, 509,
                   530, 532, 534, 573, 602, 608, 656, 660, 667, 681]

ANNOTATOR_DIRS = {
    'annotator_1': 'annotations/annotator_1',  # Adithya
    'annotator_2': 'annotations/annotator_2',  # Muhammad
    'annotator_3': 'annotations/annotator_3',  # Marlin
    'annotator_4': 'annotations/annotator_4',  # Rocky
}

ANNOTATOR_NAMES = {
    'annotator_1': 'Adithya',
    'annotator_2': 'Muhammad',
    'annotator_3': 'Marlin',
    'annotator_4': 'Rocky',
}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load(dir_path):
    """Return {trajectory_id_str: failure_classification} for all JSON in dir."""
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for f in os.listdir(dir_path):
        if not f.endswith('.json'):
            continue
        d = json.load(open(os.path.join(dir_path, f)))
        tid = str(d['trajectory_id']).replace('task_', '')
        out[tid] = d['failure_classification']
    return out


# ---------------------------------------------------------------------------
# Fleiss's kappa (no scipy)
# ---------------------------------------------------------------------------

def fleiss_kappa(matrix):
    """
    Fleiss's kappa for N subjects, each rated by the same number of raters.

    matrix: list of dicts {category: count}, one per subject.
    All rows must sum to the same n (number of raters).
    """
    if not matrix:
        return 0.0

    N = len(matrix)
    n = sum(matrix[0].values())
    if n <= 1:
        return 0.0

    categories = sorted(set(c for row in matrix for c in row))
    if len(categories) <= 1:
        return 1.0

    total = N * n
    p_j = {c: sum(row.get(c, 0) for row in matrix) / total for c in categories}
    P_e = sum(pj ** 2 for pj in p_j.values())

    P_bar = sum(
        sum(count * (count - 1) for count in row.values())
        for row in matrix
    ) / (N * n * (n - 1))

    if P_e >= 1.0:
        return 1.0

    return round((P_bar - P_e) / (1.0 - P_e), 4)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _normalise_ids(id_list):
    return set(str(t).replace('task_', '') for t in id_list)


def report_fleiss(all_anns, task_ids):
    print('=== Fleiss kappa — all 4 annotators, shared 20 ===')
    target = _normalise_ids(task_ids)
    ann_keys = list(all_anns.keys())

    rows_full = []
    per_task = []
    for t in sorted(target):
        ratings = {k: all_anns[k][t] for k in ann_keys if t in all_anns[k]}
        n_raters = len(ratings)
        if n_raters == 0:
            per_task.append((t, 0, {}, None))
            continue
        counts = {}
        for label in ratings.values():
            counts[label] = counts.get(label, 0) + 1
        agree = len(set(ratings.values())) == 1
        per_task.append((t, n_raters, ratings, agree))
        if n_raters == len(ann_keys):
            rows_full.append(counts)

    n_full = len(rows_full)
    print(f'  Target: {len(target)}  '
          f'Full coverage (all {len(ann_keys)}): {n_full}  '
          f'Partial/missing: {len(target) - n_full}')

    if n_full >= 2:
        kappa = fleiss_kappa(rows_full)
        print(f'  Fleiss kappa (full-coverage tasks only, n={n_full}): {kappa}')
    else:
        print(f'  Not enough full-coverage tasks to compute kappa (need >= 2, have {n_full}).')

    print()
    print('  Per-task:')
    name_map = ANNOTATOR_NAMES
    for t, n_raters, ratings, agree in per_task:
        if n_raters == 0:
            print(f'    task_{t}: no annotations')
            continue
        detail = '  '.join(f'{name_map.get(k, k)}={v}' for k, v in ratings.items())
        status = 'AGREE' if agree else 'DISAGREE'
        print(f'    task_{t} ({n_raters} raters): {detail}  {status}')
    print()


def report_pair(name_a, ann_a, name_b, ann_b, task_ids):
    print(f'=== Cohen kappa — {name_a} vs {name_b} ===')
    target = _normalise_ids(task_ids)
    ids_a, ids_b = set(ann_a), set(ann_b)

    both = sorted(target & ids_a & ids_b)
    only_a = sorted(target & ids_a - ids_b)
    only_b = sorted(target & ids_b - ids_a)
    neither = sorted(target - ids_a - ids_b)

    print(f'  Target: {len(target)}  '
          f'Both: {len(both)}  '
          f'Only {name_a}: {len(only_a)}  '
          f'Only {name_b}: {len(only_b)}  '
          f'Neither: {len(neither)}')

    if len(both) < 2:
        print('  Not enough overlap to compute kappa.')
        print()
        return

    labels_a = [ann_a[t] for t in both]
    labels_b = [ann_b[t] for t in both]
    kappa = compute_cohens_kappa(labels_a, labels_b)
    agree_n = sum(a == b for a, b in zip(labels_a, labels_b))
    print(f'  Cohen kappa: {kappa}  '
          f'({agree_n}/{len(both)} agree, {len(both)-agree_n}/{len(both)} disagree)')
    print()

    for t in both:
        status = 'AGREE' if ann_a[t] == ann_b[t] else 'DISAGREE'
        print(f'    task_{t}: {name_a}={ann_a[t]}  {name_b}={ann_b[t]}  {status}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_anns = {}
    for key, path in ANNOTATOR_DIRS.items():
        ann = load(path)
        all_anns[key] = ann
        print(f'{ANNOTATOR_NAMES[key]} ({key}): {len(ann)} annotations '
              f'({path})')
    print()

    # Tier 1 — Fleiss on shared 20
    report_fleiss(all_anns, SHARED_20)

    # Tier 2 — Cohen: Adithya vs Muhammad on their 40 only
    report_pair(
        'Adithya',  all_anns['annotator_1'],
        'Muhammad', all_anns['annotator_2'],
        ADITHYA_MUHAMMAD_40,
    )

    # Tier 3 — Cohen: Marlin vs Rocky on their 40 only
    report_pair(
        'Marlin', all_anns['annotator_3'],
        'Rocky',  all_anns['annotator_4'],
        MARLIN_ROCKY_40,
    )


main()
