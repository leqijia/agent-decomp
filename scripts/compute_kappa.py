"""Three-tier inter-annotator agreement.

Reports:
  failure_classification — Cohen's kappa, Gwet's AC1, raw agreement
  t_star_step             — exact match, within-1, within-3, mean abs diff

Tiers:
  Tier 1 — Fleiss's kappa across all 4 annotators on the shared 20.
  Tier 2 — Cohen + AC1 for Adithya (a1) vs Muhammad (a2) on their 40.
  Tier 3 — Cohen + AC1 for Marlin  (a3) vs Rocky    (a4) on their 40.

Writes a JSON summary to experiments/kappa_results.json so paper tables
read from a single file instead of scraping stdout.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, '.')
from harness.metrics import compute_cohens_kappa

# ---------------------------------------------------------------------------
# Annotation overlap structure (matches the actual handouts)
# ---------------------------------------------------------------------------

SHARED_20 = [28, 49, 102, 103, 104, 159, 171, 178, 181, 241,
             12, 13, 27, 62, 114, 142, 162, 169, 239, 273]

ADITHYA_MUHAMMAD_40 = [270, 277, 285, 309, 312, 351, 352, 375, 390, 393,
                       399, 411, 412, 432, 434, 442, 446, 451, 454, 460,
                       471, 473, 484, 486, 489, 494, 496, 502, 522, 528,
                       575, 594, 615, 653, 654, 655, 672, 675, 682, 688]

MARLIN_ROCKY_40 = [280, 311, 317, 324, 328, 354, 388, 391, 400, 413,
                   420, 422, 433, 437, 441, 443, 444, 447, 458, 461,
                   472, 481, 482, 483, 487, 488, 492, 507, 508, 509,
                   530, 532, 534, 573, 602, 608, 656, 660, 667, 681]

ANNOTATOR_DIRS = {
    'annotator_1': 'annotations/annotator_1',
    'annotator_2': 'annotations/annotator_2',
    'annotator_3': 'annotations/annotator_3',
    'annotator_4': 'annotations/annotator_4',
}

ANNOTATOR_NAMES = {
    'annotator_1': 'Adithya',
    'annotator_2': 'Muhammad',
    'annotator_3': 'Marlin',
    'annotator_4': 'Rocky',
}

OUT_PATH = 'experiments/kappa_results.json'


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load(dir_path):
    """Return {trajectory_id_str: {'class': str|None, 't_star': int|None}}."""
    out = {}
    if not os.path.isdir(dir_path):
        return out
    for f in os.listdir(dir_path):
        if not f.endswith('.json'):
            continue
        try:
            d = json.load(open(os.path.join(dir_path, f)))
        except json.JSONDecodeError:
            continue
        tid = str(d['trajectory_id']).replace('task_', '')
        out[tid] = {
            'class': d.get('failure_classification'),
            't_star': d.get('t_star_step'),
        }
    return out


def _normalise_ids(id_list):
    return set(str(t).replace('task_', '') for t in id_list)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def gwets_ac1(labels_a, labels_b):
    """Gwet's AC1 — robust to class-prevalence skew, unlike Cohen's kappa.

    For 2-category labels with C categories Q (the chance-agreement term):
        AC1 = (p_o - p_e) / (1 - p_e)
        p_e = 2 * pi * (1 - pi) / (C - 1) * ... -- general form below.
    """
    if len(labels_a) != len(labels_b) or len(labels_a) == 0:
        return None
    n = len(labels_a)
    cats = sorted(set(labels_a) | set(labels_b))
    C = len(cats)
    if C <= 1:
        return 1.0
    p_o = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    # marginal proportion per category, averaged across raters
    pi = {}
    for c in cats:
        pa = sum(1 for x in labels_a if x == c) / n
        pb = sum(1 for x in labels_b if x == c) / n
        pi[c] = (pa + pb) / 2.0
    p_e = sum(pi[c] * (1.0 - pi[c]) for c in cats) / (C - 1)
    if p_e >= 1.0:
        return 1.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def fleiss_kappa(matrix):
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


def t_star_stats(pairs):
    """pairs: list of (t_a, t_b) ignoring None entries."""
    valid = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not valid:
        return None
    diffs = [abs(a - b) for a, b in valid]
    n = len(diffs)
    return {
        'n': n,
        'mean_abs_diff': round(sum(diffs) / n, 3),
        'median_abs_diff': sorted(diffs)[n // 2],
        'exact_match_rate': round(sum(1 for d in diffs if d == 0) / n, 3),
        'within_1_rate':   round(sum(1 for d in diffs if d <= 1) / n, 3),
        'within_3_rate':   round(sum(1 for d in diffs if d <= 3) / n, 3),
    }


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def report_pair(name_a, ann_a, name_b, ann_b, task_ids, log):
    target = _normalise_ids(task_ids)
    both = sorted(target & set(ann_a.keys()) & set(ann_b.keys()))

    print(f'\n=== {name_a} vs {name_b}  (target {len(target)}, both annotated {len(both)}) ===')

    if len(both) < 2:
        print('  Not enough overlap.')
        log[f'{name_a}_vs_{name_b}'] = {'n': len(both)}
        return

    cls_a = [ann_a[t]['class'] for t in both]
    cls_b = [ann_b[t]['class'] for t in both]
    cls_a = ['unknown' if c is None else c for c in cls_a]
    cls_b = ['unknown' if c is None else c for c in cls_b]

    raw_agree = sum(1 for x, y in zip(cls_a, cls_b) if x == y) / len(both)
    kappa = compute_cohens_kappa(cls_a, cls_b)
    ac1 = gwets_ac1(cls_a, cls_b)

    t_pairs = [(ann_a[t]['t_star'], ann_b[t]['t_star']) for t in both]
    tstats = t_star_stats(t_pairs)

    print(f'  failure_class:  raw_agreement={raw_agree:.3f}  kappa={kappa}  AC1={ac1}')
    if tstats:
        print(f'  t_star:         exact={tstats["exact_match_rate"]}  within1={tstats["within_1_rate"]}  '
              f'within3={tstats["within_3_rate"]}  mean_abs={tstats["mean_abs_diff"]}')

    log[f'{name_a}_vs_{name_b}'] = {
        'n': len(both),
        'failure_class': {
            'raw_agreement': round(raw_agree, 4),
            'cohens_kappa': kappa,
            'gwets_ac1': ac1,
        },
        't_star': tstats,
        'tasks': both,
    }


def report_fleiss(all_anns, task_ids, log):
    print(f'\n=== Fleiss kappa (all 4 annotators, target {len(task_ids)}) ===')
    target = _normalise_ids(task_ids)
    rows = []
    coverage = []
    cls_lists = {k: [] for k in all_anns}
    for t in sorted(target):
        ratings = {k: all_anns[k][t]['class']
                   for k in all_anns if t in all_anns[k] and all_anns[k][t]['class']}
        coverage.append((t, len(ratings)))
        if len(ratings) == len(all_anns):
            counts = Counter(ratings.values())
            rows.append(dict(counts))
            for k, v in ratings.items():
                cls_lists[k].append(v)

    n_full = len(rows)
    if n_full < 2:
        print(f'  Not enough full-coverage tasks ({n_full}/{len(target)}).')
        log['fleiss_shared_20'] = {'n_full_coverage': n_full}
        return

    fk = fleiss_kappa(rows)
    print(f'  Fleiss kappa (n={n_full}): {fk}')

    # pairwise AC1 / kappa within shared set
    pair_metrics = {}
    keys = list(all_anns.keys())
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            la = []
            lb = []
            for t in sorted(target):
                if t in all_anns[ka] and t in all_anns[kb]:
                    ca = all_anns[ka][t]['class']; cb = all_anns[kb][t]['class']
                    if ca and cb:
                        la.append(ca); lb.append(cb)
            if len(la) >= 2:
                pair_metrics[f'{ANNOTATOR_NAMES[ka]}_vs_{ANNOTATOR_NAMES[kb]}'] = {
                    'n': len(la),
                    'cohens_kappa': compute_cohens_kappa(la, lb),
                    'gwets_ac1': gwets_ac1(la, lb),
                    'raw_agreement': round(sum(1 for x, y in zip(la, lb) if x == y) / len(la), 4),
                }

    log['fleiss_shared_20'] = {
        'n_full_coverage': n_full,
        'fleiss_kappa': fk,
        'pairwise_within_shared': pair_metrics,
    }
    print(f'  Pairwise within shared-20:')
    for k, v in pair_metrics.items():
        print(f'    {k}: kappa={v["cohens_kappa"]}  AC1={v["gwets_ac1"]}  raw={v["raw_agreement"]}  n={v["n"]}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_anns = {}
    for key, path in ANNOTATOR_DIRS.items():
        ann = load(path)
        all_anns[key] = ann
        print(f'{ANNOTATOR_NAMES[key]} ({key}): {len(ann)} annotations')

    log = {'overlap_structure': {
        'shared_20': SHARED_20,
        'adithya_muhammad_40': ADITHYA_MUHAMMAD_40,
        'marlin_rocky_40': MARLIN_ROCKY_40,
    }}

    report_fleiss(all_anns, SHARED_20, log)
    report_pair('Adithya', all_anns['annotator_1'],
                'Muhammad', all_anns['annotator_2'],
                ADITHYA_MUHAMMAD_40, log)
    report_pair('Marlin', all_anns['annotator_3'],
                'Rocky', all_anns['annotator_4'],
                MARLIN_ROCKY_40, log)

    # Cross-pair sanity checks (smaller, supplementary)
    report_pair('Adithya', all_anns['annotator_1'],
                'Marlin', all_anns['annotator_3'],
                SHARED_20, log)
    report_pair('Muhammad', all_anns['annotator_2'],
                'Rocky', all_anns['annotator_4'],
                SHARED_20, log)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    json.dump(log, open(OUT_PATH, 'w'), indent=2)
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()
