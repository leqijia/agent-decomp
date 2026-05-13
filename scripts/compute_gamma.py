"""Compute Gamma(L) — context gap by trajectory length bin.

Default inputs are the current per-condition results artifacts:

  ORACLE_RESULTS  experiments/exp3/oracle_external/results.json
  RAW_RESULTS     experiments/exp3/raw/results.json

Gamma(L) = Acc_oracle(L) - Acc_raw(L) per length bin.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import LENGTH_BINS, compute_gamma_L

ORACLE_RESULTS = 'experiments/exp3/oracle_external/results.json'
RAW_RESULTS    = 'experiments/exp3/raw/results.json'
OUT_PATH   = 'experiments/gamma_results.json'

BINS = [(0, 10), (11, 20), (21, 40), (41, 80)]


def bin_for_length(L):
    for lo, hi in BINS:
        if lo <= L <= hi:
            return f'{lo}-{hi}'
    print(f'  WARNING: trajectory length {L} is out-of-spec (>80), logging as "80+"')
    return '80+'


def load_results(csv_path):
    """Load CSV into list of {task_id, trajectory_length, success} dicts."""
    if not os.path.exists(csv_path):
        print(f'  WARNING: {csv_path} not found — returning empty list.')
        return []
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    'task_id':           row['task_id'],
                    'trajectory_length': int(row['trajectory_length']),
                    'success':           int(row['success']),
                })
            except (KeyError, ValueError) as e:
                print(f'  WARNING: skipping malformed row {dict(row)}: {e}')
    return rows


def load_results_json(path):
    """Load modern results.json payload into harness.metrics-compatible rows."""
    if not os.path.exists(path):
        print(f'  WARNING: {path} not found — returning empty list.')
        return []
    payload = json.load(open(path))
    return payload.get('results', [])


def compute_gamma_from_json(oracle_results, raw_results):
    """Return compute_gamma_L output, with empty bins included for stable tables."""
    gamma = compute_gamma_L(oracle_results, raw_results)
    return {
        b: gamma.get(b, {
            'oracle_acc': 0.0,
            'raw_acc': 0.0,
            'gamma': 0.0,
            'count': 0,
        })
        for b in LENGTH_BINS
    }


def compute_gamma(oracle_rows, raw_rows):
    """Return {bin_name: {oracle_acc, raw_acc, gamma, oracle_n, raw_n}}."""
    def bin_results(rows):
        binned = {}
        for r in rows:
            b = bin_for_length(r['trajectory_length'])
            if b is not None:
                binned.setdefault(b, []).append(r['success'])
        return binned

    oracle_bins = bin_results(oracle_rows)
    raw_bins    = bin_results(raw_rows)

    bin_names = [f'{lo}-{hi}' for lo, hi in BINS] + ['80+']
    out = {}
    for b in bin_names:
        o = oracle_bins.get(b, [])
        r = raw_bins.get(b, [])
        o_acc = round(sum(o) / len(o), 4) if o else None
        r_acc = round(sum(r) / len(r), 4) if r else None
        gamma = round(o_acc - r_acc, 4) if (o_acc is not None and r_acc is not None) else None
        out[b] = {
            'oracle_acc': o_acc,
            'raw_acc':    r_acc,
            'gamma':      gamma,
            'oracle_n':   len(o),
            'raw_n':      len(r),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='Compute Gamma(L).')
    parser.add_argument('--oracle-results', default=ORACLE_RESULTS)
    parser.add_argument('--raw-results', default=RAW_RESULTS)
    parser.add_argument('--out', default=OUT_PATH)
    parser.add_argument(
        '--legacy-csv',
        action='store_true',
        help='Read legacy CSV files with task_id,trajectory_length,success columns.',
    )
    args = parser.parse_args()

    print(f'Loading oracle results: {args.oracle_results}')
    if args.legacy_csv:
        oracle_rows = load_results(args.oracle_results)
    else:
        oracle_rows = load_results_json(args.oracle_results)
    print(f'  {len(oracle_rows)} rows')

    print(f'Loading raw results:    {args.raw_results}')
    if args.legacy_csv:
        raw_rows = load_results(args.raw_results)
    else:
        raw_rows = load_results_json(args.raw_results)
    print(f'  {len(raw_rows)} rows')
    print()

    results = (
        compute_gamma(oracle_rows, raw_rows)
        if args.legacy_csv
        else compute_gamma_from_json(oracle_rows, raw_rows)
    )

    header = f'{"Bin":<10}  {"Oracle acc":>10}  {"Raw acc":>8}  {"Gamma":>6}  {"N":>6}'
    print('=== Gamma(L) ===')
    print(header)
    print('-' * len(header))
    for b, v in results.items():
        o = f'{v["oracle_acc"]:.3f}' if v['oracle_acc'] is not None else 'N/A'
        r = f'{v["raw_acc"]:.3f}'    if v['raw_acc']    is not None else 'N/A'
        g = f'{v["gamma"]:.3f}'      if v['gamma']      is not None else 'N/A'
        n = v.get('count')
        if n is None:
            n = max(v.get('oracle_n', 0), v.get('raw_n', 0))
        print(f'{b:<10}  {o:>10}  {r:>8}  {g:>6}  {n:>6}')

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nSaved to {args.out}')


if __name__ == '__main__':
    main()
