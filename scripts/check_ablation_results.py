import json
import os

results_dir = 'experiments/ablations'
files = sorted(f for f in os.listdir(results_dir) if f.endswith('.json'))
print(f'Total results: {len(files)}')
for f in files:
    d = json.load(open(os.path.join(results_dir, f)))
    success = d.get('success')
    stop = d.get('stop_reason')
    print(f'{f}: success={success} stop={stop}')
