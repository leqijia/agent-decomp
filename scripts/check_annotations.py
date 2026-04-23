import json
import os

dirs = [
    ('ANNOTATOR_1', 'annotations/annotator_1'),
    ('ANNOTATOR_2', 'annotations/annotator_2'),
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
