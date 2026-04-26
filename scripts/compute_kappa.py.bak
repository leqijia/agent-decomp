import json
import os
import sys
sys.path.insert(0, '.')
from harness.metrics import compute_cohens_kappa

ann1_dir = 'annotations/annotator_1'
ann2_dir = 'annotations/annotator_2'


def load(dir_path):
    out = {}
    for f in os.listdir(dir_path):
        if f.endswith('.json'):
            d = json.load(open(os.path.join(dir_path, f)))
            out[d['trajectory_id']] = d['failure_classification']
    return out


a1 = load(ann1_dir)
a2 = load(ann2_dir)

overlap = sorted(set(a1.keys()) & set(a2.keys()))
print(f'annotator_1 annotations: {len(a1)}')
print(f'annotator_2 annotations: {len(a2)}')
print(f'Overlapping trajectories: {len(overlap)}')
print(f'Overlap: {overlap}')

if len(overlap) < 2:
    print('Not enough overlap to compute kappa yet.')
else:
    labels1 = [a1[t] for t in overlap]
    labels2 = [a2[t] for t in overlap]
    kappa = compute_cohens_kappa(labels1, labels2)
    print(f'Cohen kappa: {kappa}')
    for t in overlap:
        match = 'AGREE' if a1[t] == a2[t] else 'DISAGREE'
        print(f'  {t}: annotator_1={a1[t]} annotator_2={a2[t]} {match}')
