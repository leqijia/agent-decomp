"""Compute Gamma(L) — context gap by trajectory length bin.

Reads two CSV files with columns: task_id, trajectory_length, success (0 or 1).

  ORACLE_CSV  experiments/oracle_results.csv
  RAW_CSV     experiments/raw_results.csv

Gamma(L) = Acc_oracle(L) - Acc_raw(L) per length bin.

Four bins per Section 2.4 of the project proposal:
  0-10, 11-20, 21-40, 41-80   (trajectories >80 steps logged as "80+")
"""
import csv
import json
import os
import sys

ORACLE_CSV = 'experiments/oracle_results.csv'
RAW_CSV    = 'experiments/raw_results.csv'
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
    print(f'Loading oracle results: {ORACLE_CSV}')
    oracle_rows = load_results(ORACLE_CSV)
    print(f'  {len(oracle_rows)} rows')

    print(f'Loading raw results:    {RAW_CSV}')
    raw_rows = load_results(RAW_CSV)
    print(f'  {len(raw_rows)} rows')
    print()

    results = compute_gamma(oracle_rows, raw_rows)

    header = f'{"Bin":<10}  {"Oracle acc":>10}  {"Raw acc":>8}  {"Gamma":>6}  {"N oracle":>8}  {"N raw":>6}'
    print('=== Gamma(L) ===')
    print(header)
    print('-' * len(header))
    for b, v in results.items():
        o = f'{v["oracle_acc"]:.3f}' if v['oracle_acc'] is not None else 'N/A'
        r = f'{v["raw_acc"]:.3f}'    if v['raw_acc']    is not None else 'N/A'
        g = f'{v["gamma"]:.3f}'      if v['gamma']      is not None else 'N/A'
        print(f'{b:<10}  {o:>10}  {r:>8}  {g:>6}  {v["oracle_n"]:>8}  {v["raw_n"]:>6}')

    json.dump(results, open(OUT_PATH, 'w'), indent=2)
    print(f'\nSaved to {OUT_PATH}')


main()
