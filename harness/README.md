# Evaluation Harness

This module provides a standardized way to run and measure agent performance
across all experimental conditions. See proposal Section 3 for full details.

## What it does

Ensures every condition (raw, oracle, self_generated, env_only, sliding_window,
observation_masking) is run and measured the same way so results are comparable
across experiments and teammates.

## Functions

### run_condition(trajectory_path, condition, task_id) -> dict
Runs a single trajectory under a given condition.
STUB - pending Rocky's interface spec.

### evaluate_batch(trajectory_dir, condition) -> list[dict]
Runs run_condition on all trajectories in a directory.
STUB - pending Rocky's interface spec.

### compute_metrics(results) -> dict
Takes a list of run_condition outputs and returns success_rate and avg_tokens.
Fully implemented.

## Usage

### Test the harness
python harness/test_harness.py

### Import compute_metrics in your own script
from harness.evaluator import compute_metrics

results = [
    {"task_id": "task_001", "condition": "raw", "success": True, "tokens_used": 3200},
    {"task_id": "task_002", "condition": "raw", "success": False, "tokens_used": 4100},
]
print(compute_metrics(results))

## Status
- compute_metrics: implemented and tested
- run_condition: stub, pending Rocky's interface spec
- evaluate_batch: stub, pending Rocky's interface spec

## Who to contact
Marlin - harness structure and metrics
Rocky  - run_condition implementation, interface spec
