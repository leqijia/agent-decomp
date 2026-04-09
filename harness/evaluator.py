import os
import sys
import json

# baselines lives under oracle/ — add it to path so "baselines.trajectory" resolves
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "oracle"))

from baselines.trajectory import (
    sliding_window_truncate,
    mask_observations,
    serialize_trajectory,
    count_tokens,
)

VALID_CONDITIONS = {
    "raw",
    "oracle",
    "self_generated",
    "env_only",
    "sliding_window",
    "observation_masking",
}


def run_condition(trajectory_path, condition, task_id, window_size=4000):
    """
    Run a single trajectory under a given experimental condition.

    Args:
        trajectory_path: path to the trajectory .json file
        condition: one of VALID_CONDITIONS
        task_id: string identifier for the task
        window_size: max tokens for sliding_window condition (4000, 8000, or 16000)

    Returns:
        {"task_id": str, "condition": str, "success": None, "tokens_used": int, "dropped_steps": int}
    """
    with open(trajectory_path) as f:
        data = json.load(f)
    steps = data["steps"]

    if condition == "sliding_window":
        result = sliding_window_truncate(steps, max_tokens=window_size)
        tokens_used = result.trajectory_token_count
        dropped_steps = result.dropped_prefix_steps

    elif condition == "observation_masking":
        masked = mask_observations(steps)
        serialized = serialize_trajectory(masked)
        tokens_used = count_tokens(serialized)
        dropped_steps = 0

    elif condition in ("raw", "oracle", "self_generated", "env_only"):
        raise NotImplementedError("Pending Rocky's interface spec")

    else:
        raise ValueError(f"Unknown condition: {condition!r}. Must be one of {VALID_CONDITIONS}")

    return {
        "task_id": task_id,
        "condition": condition,
        "success": None,
        "tokens_used": tokens_used,
        "dropped_steps": dropped_steps,
    }


def evaluate_batch(trajectory_dir, condition):
    """
    Run run_condition on all .json files in trajectory_dir.

    Args:
        trajectory_dir: path to directory containing trajectory .json files
        condition: one of VALID_CONDITIONS

    Returns:
        list of run_condition result dicts
    """
    raise NotImplementedError("Pending Rocky's interface spec")


def compute_metrics(results):
    """
    Compute aggregate metrics from a list of run_condition outputs.

    Args:
        results: list of dicts, each with keys: task_id, condition, success, tokens_used

    Returns:
        {"success_rate": float, "avg_tokens": float}
    """
    if not results:
        return {"success_rate": 0.0, "avg_tokens": 0.0}

    scored = [r for r in results if r["success"] is not None]
    if scored:
        success_rate = round(sum(1 for r in scored if r["success"]) / len(scored), 3)
    else:
        success_rate = 0.0

    avg_tokens = round(sum(r["tokens_used"] for r in results) / len(results), 1)

    return {"success_rate": success_rate, "avg_tokens": avg_tokens}
