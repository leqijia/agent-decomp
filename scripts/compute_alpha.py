import json
import os
import sys
sys.path.insert(0, '.')
from harness.metrics import compute_alpha_decomposition

BASELINE_DIR = 'experiments/oracle_baseline'
STALE_TASKS = ['322', '332', '282', '308', '327', '342', '349', '355']

oracle_results = []
envonly_results = []
excluded = []

for f in sorted(os.listdir(BASELINE_DIR)):
    if not f.endswith('.json'):
        continue
    # Check if stale
    task_id = f.split('_')[0]
    if task_id in STALE_TASKS:
        excluded.append(f)
        continue
    d = json.load(open(os.path.join(BASELINE_DIR, f)))
    if d.get('success') is None:
        excluded.append(f + ' (success=None)')
        continue
    if 'envonly' in f:
        envonly_results.append(d)
    elif 'oracle' in f:
        oracle_results.append(d)

print('=== Included ===')
print(f'Oracle: {len(oracle_results)} files')
print(f'Envonly: {len(envonly_results)} files')
print()
print('=== Excluded ===')
for e in excluded:
    print(f'  {e}')
print()

if len(oracle_results) == 0 or len(envonly_results) == 0:
    print('Not enough results to compute alpha decomposition.')
    sys.exit(1)

result = compute_alpha_decomposition(oracle_results, envonly_results)

print('=== Alpha Decomposition ===')
print(f'R_full (oracle recovery):   {result["R_full"]:.3f}')
print(f'R_env  (env-only recovery): {result["R_env"]:.3f}')
print()
print(f'alpha_context:    {result["alpha_context"]:.3f}  95% CI: {result["ci_95"]["alpha_context"]}')
print(f'alpha_env:        {result["alpha_env"]:.3f}  95% CI: {result["ci_95"]["alpha_env"]}')
print(f'alpha_capability: {result["alpha_capability"]:.3f}  95% CI: {result["ci_95"]["alpha_capability"]}')
print(f'Sum: {result["alpha_context"] + result["alpha_env"] + result["alpha_capability"]:.3f}')

out_path = 'experiments/alpha_results.json'
json.dump(result, open(out_path, 'w'), indent=2)
print(f'\nSaved to {out_path}')
