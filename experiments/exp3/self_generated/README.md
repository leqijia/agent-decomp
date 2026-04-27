# Experiment 3 — Self-generated state condition

This directory holds the manifest and metrics view for **Experiment 3, condition 2: self-generated state** (per the MMLA proposal §2.5: *"the agent maintains a structured state summary it generates and updates itself"*).

## What this directory is

A first-class run of the [`StateAct` agent](../../../agent/stateact_agent.py) over the same task suite that backs Exp3 raw. Trajectories are produced by `scripts/run_condition_batch.py --condition self_generated` and land in `experiments/exp3/self_generated/data/`. The manifest and metrics are then sidecars over that directory, exactly mirroring the layout used by [`experiments/exp3/raw/`](../raw/README.md).

The task suite is taken from [`experiments/exp3/raw/manifest.csv`](../raw/manifest.csv) so that across-condition comparisons (Γ(L), δ_synth(L)) are computed on identical task ids.

## Files

| File           | Producer                                            | Purpose                                                                                                                                                                                                                                |
| -------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/`        | `scripts/run_condition_batch.py --condition self_generated` | Per-task trajectory JSON written by `harness.evaluator.run_episode("self_generated", …)`. One file per task id; schema defined by [`trajectories/SPEC.md`](../../../trajectories/SPEC.md).                                              |
| `manifest.csv` | `scripts/build_condition_manifest.py`               | One row per trajectory. Columns: `task_id, trajectory_path, sites, model, total_steps, length_bin, stop_reason, success, eval_score, condition, experiment_id`. Schema-compatible with the raw manifest.                              |
| `results.json` | `scripts/build_condition_manifest.py`               | List of result dicts in the shape `harness.metrics.compute_alpha_decomposition` / `compute_gamma_L` and `compute_delta_synth` consume. Input for the cross-condition deltas.                                                            |
| `skipped.csv`  | `scripts/build_condition_manifest.py`               | Sidecar listing trajectories with pipeline-anomaly `stop_reason` (`parse_failures`, `crash`, `replay_desync`). Rows are still present in `manifest.csv` / `results.json`; `skipped.csv` is a filter handle, not an exclusion list.     |
| `metrics.json` | `scripts/compute_condition_metrics.py`              | Aggregate success rates (overall, by length bin, by site, by stop reason) with 95% Wilson confidence intervals, plus `tokens` and `eval_score` summaries that feed into the baselines table.                                            |

## How to (re)generate

```bash
python scripts/run_condition_batch.py --condition self_generated \
    --task-list experiments/exp3/raw/manifest.csv \
    --out-dir   experiments/exp3/self_generated/data \
    --model     qwen/qwen3.5-27b

python scripts/build_condition_manifest.py \
    --data-dir experiments/exp3/self_generated/data \
    --out-dir  experiments/exp3/self_generated \
    --condition self_generated --experiment-id exp3

python scripts/compute_condition_metrics.py \
    --results experiments/exp3/self_generated/results.json \
    --out     experiments/exp3/self_generated/metrics.json
```

## Notes

- **Agent**: `agent/stateact_agent.py`, dispatched via `harness.conditions.CONDITION_REGISTRY["self_generated"]`. Default `max_state_history=5` (state-summary truncation window); change in the agent class if the experimental design needs a different default.
- **Model deviation**: this suite uses `qwen/qwen3.5-27b` (matching the Exp3 raw collection) rather than the GPT-4-class model named in the proposal §2.5; cross-condition comparisons are still apples-to-apples because raw and self-generated runs share the same model.
- **Cross-condition contract**: `task_id` set in this manifest must equal the `task_id` set in `experiments/exp3/raw/manifest.csv`. Any divergence (e.g. a task that crashed under self_generated but not raw) shows up as an unscored row, never as a missing row.
