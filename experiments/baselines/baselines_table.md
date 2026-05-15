# Baselines table

_Generated: 2026-05-09T04:05:32Z_

## Headline (overall)

| Method | n | Success rate | 95% CI | Avg tokens / task | Avg eval_score |
|---|---:|---:|---|---:|---:|
| raw | 276 | 23.5% | [18.9%, 28.9%] | 104,690 | 0.2355 |
| self_generated | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |
| sliding_window | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |
| observation_masking | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |
| acon | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |
| agentdiet | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |
| perfect_retrieval | 0 | 0.0% | [0.0%, 0.0%] | 0 | n/a |

## By length bin (success rate)

| Method | <=10 | 11-20 | 21-40 | 41-80 |
|---|---:|---:|---:|---:|
| raw | 37.2% | 4.5% | 6.7% | 2.7% |
| self_generated | 0.0% | 0.0% | 0.0% | 0.0% |
| sliding_window | 0.0% | 0.0% | 0.0% | 0.0% |
| observation_masking | 0.0% | 0.0% | 0.0% | 0.0% |
| acon | 0.0% | 0.0% | 0.0% | 0.0% |
| agentdiet | 0.0% | 0.0% | 0.0% | 0.0% |
| perfect_retrieval | 0.0% | 0.0% | 0.0% | 0.0% |

## Notes

- `n` is the count of *scored* tasks (tasks where evaluation succeeded; crashes are reported separately as unscored in the per-condition metrics.json).
- 95% CIs are Wilson-score intervals on the per-method success rate (`harness.metrics._wilson_ci`).
- `Avg tokens / task` sums prompt + completion tokens across every step in every trajectory and divides by tasks.
- `perfect_retrieval` uses live-time query semantics (`intent + current_observation`), not the strict-oracle query from `baselines/perfect_retrieval.py`. See `experiments/baselines/README.md`.
