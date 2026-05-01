"""Diagnostic: t_star_step agreement across annotators.

Reports spread (max - min) of t* values for trajectories annotated by
2+ annotators, and flags any t*=0 entries (unsolvable-from-start).
"""
import json
import os

ANNOTATOR_DIRS = {
    'annotator_1': ('annotations/annotator_1', 'Adithya'),
    'annotator_2': ('annotations/annotator_2', 'Muhammad'),
    'annotator_3': ('annotations/annotator_3', 'Marlin'),
    'annotator_4': ('annotations/annotator_4', 'Rocky'),
}

SHARED_20 = {28, 49, 102, 103, 104, 159, 171, 178, 181, 241,
             12, 13, 27, 62, 114, 142, 162, 169, 239, 273}

ADITHYA_MUHAMMAD_40 = {270, 277, 285, 309, 312, 351, 352, 375, 390, 393,
                       399, 411, 412, 432, 434, 442, 446, 451, 454, 460,
                       471, 473, 484, 486, 489, 494, 496, 502, 522, 528,
                       575, 594, 615, 653, 654, 655, 672, 675, 682, 688}

MARLIN_ROCKY_40 = {280, 311, 317, 324, 328, 354, 388, 391, 400, 413,
                   420, 422, 433, 437, 441, 443, 444, 447, 458, 461,
                   472, 481, 482, 483, 487, 488, 492, 507, 508, 509,
                   530, 532, 534, 573, 602, 608, 656, 660, 667, 681}


def batch_label(tid_int):
    if tid_int in SHARED_20:
        return 'shared'
    if tid_int in ADITHYA_MUHAMMAD_40:
        return 'AM-pair'
    if tid_int in MARLIN_ROCKY_40:
        return 'MR-pair'
    return 'other'


def load_annotations():
    """Return {tid_str: {annotator_key: t_star_step}}."""
    data = {}
    for key, (path, name) in ANNOTATOR_DIRS.items():
        if not os.path.isdir(path):
            print(f'  WARNING: {path} not found, skipping {name}')
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith('.json'):
                continue
            try:
                d = json.load(open(os.path.join(path, fname)))
            except (json.JSONDecodeError, OSError) as e:
                print(f'  WARNING: skipping {fname}: {e}')
                continue
            raw_id = str(d.get('trajectory_id', '')).replace('task_', '')
            t_star = d.get('t_star_step')
            if t_star is None:
                continue
            data.setdefault(raw_id, {})[key] = t_star
    return data


def spread_bucket(spread):
    if spread == 0:
        return 'exact'
    if spread <= 2:
        return 'within-2'
    if spread <= 5:
        return '3-to-5'
    return 'wider-5'


def fmt(val):
    return str(val) if val is not None else '-'


def main():
    all_anns = load_annotations()

    ann_order = ['annotator_1', 'annotator_2', 'annotator_3', 'annotator_4']
    ann_short = {'annotator_1': 'Adithya', 'annotator_2': 'Muhammad',
                 'annotator_3': 'Marlin',  'annotator_4': 'Rocky'}

    # Only rows with 2+ annotators
    multi = {tid: anns for tid, anns in all_anns.items() if len(anns) >= 2}

    # Compute spreads
    rows = []
    for tid, anns in multi.items():
        vals = list(anns.values())
        spread = max(vals) - min(vals)
        try:
            tid_int = int(tid)
        except ValueError:
            tid_int = -1
        batch = batch_label(tid_int)
        rows.append((tid, anns, spread, batch))

    rows.sort(key=lambda r: (-r[2], r[0]))

    # ── Per-trajectory table ──────────────────────────────────────────────────
    col_w = 9
    header = (f'{"task":<10} {"adithya":>{col_w}} {"muhammad":>{col_w}} '
              f'{"marlin":>{col_w}} {"rocky":>{col_w}} {"spread":>6}  batch')
    print(header)
    print('-' * len(header))
    for tid, anns, spread, batch in rows:
        a = fmt(anns.get('annotator_1'))
        m = fmt(anns.get('annotator_2'))
        ma = fmt(anns.get('annotator_3'))
        r = fmt(anns.get('annotator_4'))
        print(f'task_{tid:<5} {a:>{col_w}} {m:>{col_w}} '
              f'{ma:>{col_w}} {r:>{col_w}} {spread:>6}  {batch}')

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(rows)
    print()
    print(f'Total trajectories with 2+ annotators: {n}')

    if n == 0:
        print('  (no data)')
        return

    buckets = {'exact': 0, 'within-2': 0, '3-to-5': 0, 'wider-5': 0}
    spreads = []
    for _, _, spread, _ in rows:
        buckets[spread_bucket(spread)] += 1
        spreads.append(spread)

    def pct(x):
        return f'{100 * x / n:.1f}%'

    print(f'  Exact agreement on t*: {buckets["exact"]} ({pct(buckets["exact"])})')
    print(f'  Within 2 steps:        {buckets["within-2"]} ({pct(buckets["within-2"])})')
    print(f'  Within 3-5 steps:      {buckets["3-to-5"]} ({pct(buckets["3-to-5"])})')
    print(f'  Wider than 5 steps:    {buckets["wider-5"]} ({pct(buckets["wider-5"])})')

    spreads_sorted = sorted(spreads)
    mid = len(spreads_sorted) // 2
    median = (spreads_sorted[mid] if len(spreads_sorted) % 2 == 1
              else (spreads_sorted[mid - 1] + spreads_sorted[mid]) / 2)
    print()
    print(f'  Mean spread:   {sum(spreads) / len(spreads):.1f}')
    print(f'  Median spread: {median}')
    print(f'  Max spread:    {max(spreads)}')

    # ── t*=0 section ─────────────────────────────────────────────────────────
    zeros = []
    for tid, anns in all_anns.items():
        zero_by = [k for k, v in anns.items() if v == 0]
        if zero_by:
            zeros.append((tid, anns, zero_by))
    zeros.sort(key=lambda r: r[0])

    print()
    print(f't*=0 entries (task marked unsolvable from start): {len(zeros)}')
    if zeros:
        for tid, anns, zero_by in zeros:
            who = ', '.join(ann_short[k] for k in zero_by)
            others = {ann_short[k]: v for k, v in anns.items()
                      if k not in zero_by}
            others_str = '  '.join(f'{n}={v}' for n, v in others.items()) or '(no other annotations)'
            print(f'  task_{tid}: t*=0 by [{who}]  others: {others_str}')


if __name__ == '__main__':
    main()
