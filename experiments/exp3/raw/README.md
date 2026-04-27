# Experiment 3 — Raw condition

This directory holds the manifest and metrics view for **Experiment 3, condition 1: raw trajectory** (per the MMLA proposal §2.5: *"the agent receives its full unmodified trajectory τ_{1:t} at each step"*).

## What this directory is

A *labeling* of the existing trajectory collection in `trajectories/data/` as the Experiment 3 raw condition. The trajectory JSON files themselves are **not copied or modified** — they are referenced by path from `manifest.csv`.

## Files

| File           | Producer                                       | Purpose                                                                                                |
| -------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `manifest.csv` | `scripts/build_exp3_raw_manifest.py`           | One row per trajectory. Columns: `task_id, trajectory_path, sites, model, total_steps, length_bin, stop_reason, success, eval_score, condition, experiment_id`. Defines the Exp3 task suite. |
| `results.json` | `scripts/build_exp3_raw_manifest.py`           | List of result dicts in the shape `harness.metrics.compute_alpha_decomposition` / `compute_gamma_L` consume. Input for downstream comparisons. |
| `skipped.csv`  | `scripts/build_exp3_raw_manifest.py`           | Sidecar listing trajectories with pipeline-anomaly `stop_reason` (`parse_failures`, `crash`, `replay_desync`). These rows are **still present** in `manifest.csv` and `results.json` (counted as `success=False` or null) — `skipped.csv` is a filter handle, not an exclusion list. |
| `metrics.json` | `scripts/compute_exp3_raw_metrics.py`          | Aggregate success rates (overall, by length bin, by site, by stop reason) with 95% Wilson confidence intervals. Canonical input for `Γ(L)` plots. |

