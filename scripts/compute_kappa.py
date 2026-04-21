import json
import os
import sys
sys.path.insert(0, '.')
from harness.metrics import compute_cohens_kappa

adithya_dir = 'annotator/annotations/Adithya'
muhammad_dir = 'annotations/Muhammad'

adithya = {}
for f in os.listdir(adithya_dir):
    if f.endswith('.json'):
        d = json.load(open(os.path.join(adithya_dir, f)))
        tid = d['trajectory_id']
        adithya[tid] = d['failure_classification']

muhammad = {}
for f in os.listdir(muhammad_dir):
    if f.endswith('.json'):
        d = json.load(open(os.path.join(muhammad_dir, f)))
        tid = d['trajectory_id']
        muhammad[tid] = d['failure_classification']

overlap = sorted(set(adithya.keys()) & set(muhammad.keys()))
print(f'Adithya annotations: {len(adithya)}')
print(f'Muhammad annotations: {len(muhammad)}')
print(f'Overlapping trajectories: {len(overlap)}')
print(f'Overlap: {overlap}')

if len(overlap) < 2:
    print('Not enough overlap to compute kappa yet.')
else:
    labels_a = [adithya[t] for t in overlap]
    labels_m = [muhammad[t] for t in overlap]
    kappa = compute_cohens_kappa(labels_a, labels_m)
    print(f'Cohen kappa: {kappa}')
    for t in overlap:
        match = 'AGREE' if adithya[t] == muhammad[t] else 'DISAGREE'
        print(f'  {t}: Adithya={adithya[t]} Muhammad={muhammad[t]} {match}')
