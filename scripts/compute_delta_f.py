import json
import os
import sys
sys.path.insert(0, '.')

BASELINE_DIR = 'experiments/oracle_baseline'
ABLATION_DIR = 'experiments/ablations'
RESULTS_DIR = 'experiments'

# Load full oracle baseline results per task
baseline = {}
for f in os.listdir(BASELINE_DIR):
    if not f.endswith('_oracle_result.json'):
        continue
    task_id = f.replace('_tstar_oracle_result.json', '')
    d = json.load(open(os.path.join(BASELINE_DIR, f)))
    baseline[task_id] = d.get('success')

print(f'Baseline results loaded: {len(baseline)} trajectories')
print(f'Full oracle recovery rate: {sum(1 for v in baseline.values() if v) / len(baseline):.3f}')
print()

# Load ablation results per task per field
fields = ['g', 'P_t', 'R_t', 'e_t', 'C', 'F_t', 'K_t']
ablation_results = {f: {} for f in fields}

for fname in os.listdir(ABLATION_DIR):
    if not fname.endswith('_result.json'):
        continue
    # parse filename: 114_t17_ablate_F_t_result.json
    parts = fname.replace('_result.json', '').split('_ablate_')
    if len(parts) != 2:
        continue
    task_part = parts[0]  # e.g. 114_t17
    field = parts[1]      # e.g. F_t
    task_id = task_part.split('_')[0]  # e.g. 114
    if field not in fields:
        continue
    d = json.load(open(os.path.join(ABLATION_DIR, fname)))
    ablation_results[field][task_id] = d.get('success')

# Compute Delta_f per field
print('=== Ablation Results (Delta_f) ===')
print(f'{"Field":<8} {"Ablated acc":>12} {"Full oracle acc":>16} {"Delta_f":>10} {"N":>5}')
print('-' * 55)

delta_f = {}
for field in fields:
    ablated = ablation_results[field]
    # Only use tasks that have both baseline and ablation results
    common = [t for t in ablated if t in baseline
              and ablated[t] is not None
              and baseline[t] is not None]
    if not common:
        print(f'{field:<8} {"N/A":>12} {"N/A":>16} {"N/A":>10} {0:>5}')
        continue
    acc_ablated = sum(1 for t in common if ablated[t]) / len(common)
    acc_full = sum(1 for t in common if baseline[t]) / len(common)
    delta = acc_full - acc_ablated
    delta_f[field] = delta
    print(f'{field:<8} {acc_ablated:>12.3f} {acc_full:>16.3f} {delta:>10.3f} {len(common):>5}')

print()
print('=== Field Importance Ranking ===')
for field, delta in sorted(delta_f.items(), key=lambda x: x[1], reverse=True):
    bar = '#' * int(delta * 20)
    print(f'{field:<8} {delta:.3f} {bar}')

# Save results
output = {
    'baseline_recovery_rate': sum(1 for v in baseline.values() if v) / len(baseline) if baseline else None,
    'n_trajectories': len(baseline),
    'delta_f': delta_f,
    'field_ranking': sorted(delta_f.keys(), key=lambda x: delta_f[x], reverse=True),
}
out_path = os.path.join(RESULTS_DIR, 'delta_f_results.json')
json.dump(output, open(out_path, 'w'), indent=2)
print(f'\nSaved to {out_path}')
