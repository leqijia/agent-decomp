import json, os, shutil

data_dir = 'trajectories/data'
copies_dir = 'trajectories/oracle_copies'
existing = set(os.listdir(copies_dir))
copied = []

for f in sorted(os.listdir(data_dir)):
    try:
        t = json.load(open(os.path.join(data_dir, f)))
        steps = t.get('total_steps', 0)
        if steps >= 15 and f not in existing:
            shutil.copy(os.path.join(data_dir, f), os.path.join(copies_dir, f))
            copied.append((f, steps))
            print(f'Copied {f}: {steps} steps')
    except Exception as e:
        print(f'SKIP {f}: {e}')

print(f'\nTotal copied: {len(copied)}')
print(f'Total in oracle_copies: {len(os.listdir(copies_dir))}')
