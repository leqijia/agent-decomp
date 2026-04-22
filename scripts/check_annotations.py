import json
import os

dirs = [
    ('ADITHYA', 'annotations/Adithya'),
    ('MUHAMMAD', 'annotations/Muhammad'),
]

for name, d in dirs:
    print(f'=== {name} ===')
    if not os.path.exists(d):
        print(f'  Directory not found: {d}')
        continue
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        data = json.load(open(os.path.join(d, f)))
        print(f'  {f}: {json.dumps(data, indent=2)}')
    print()
