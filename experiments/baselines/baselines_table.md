# Baselines table

_Generated: 2026-05-17T00:25:34Z_

## Headline (overall)

| Method | n | Success rate | 95% CI | Avg tokens / task | Avg eval_score |
|---|---:|---:|---|---:|---:|
| raw | 276 | 23.5% | [18.9%, 28.9%] | 104,061 | 0.2355 |
| self_generated | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |
| sliding_window | 229 | 39.7% | [33.6%, 46.2%] | 63,105 | 0.3974 |
| observation_masking | 243 | 39.1% | [33.2%, 45.4%] | 88,743 | 0.3909 |
| acon | 76 | 44.7% | [34.1%, 55.9%] | 8,778 | 0.4474 |
| agentdiet | 29 | 44.8% | [28.4%, 62.5%] | 5,674 | 0.4483 |
| perfect_retrieval | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |

## By length bin (success rate)

| Method | <=10 | 11-20 | 21-40 | 41-80 |
|---|---:|---:|---:|---:|
| raw | 37.2% | 4.5% | 6.7% | 2.7% |
| self_generated | 0.0% | 0.0% | 0.0% | 0.0% |
| sliding_window | 45.2% | 20.8% | 66.7% | 21.2% |
| observation_masking | 47.6% | 33.3% | 22.2% | 14.0% |
| acon | 49.2% | 16.7% | 0.0% | 0.0% |
| agentdiet | 44.4% | 50.0% | 0.0% | 0.0% |
| perfect_retrieval | 0.0% | 0.0% | 0.0% | 0.0% |

## Notes

- `n` is the count of *scored* tasks (tasks where evaluation succeeded; crashes are reported separately as unscored in the per-condition metrics.json).
- 95% CIs are Wilson-score intervals on the per-method success rate (`harness.metrics._wilson_ci`).
- `Avg tokens / task` sums prompt + completion tokens across every step in every trajectory and divides by tasks.
- `perfect_retrieval` uses live-time query semantics (`intent + current_observation`), not the strict-oracle query from `baselines/perfect_retrieval.py`. See `experiments/baselines/README.md`.
