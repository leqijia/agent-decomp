import os
import json

VALID_CONDITIONS = {
    "raw",
    "oracle",
    "self_generated",
    "env_only",
    "sliding_window",
    "observation_masking",
}


def run_condition(trajectory_path, condition, task_id):
    """
    Run a single trajectory under a given experimental condition.

    Args:
        trajectory_path: path to the trajectory .json file
        condition: one of VALID_CONDITIONS
        task_id: string identifier for the task

    Returns:
        {"task_id": str, "condition": str, "success": bool, "tokens_used": int}
    """
    raise NotImplementedError("Pending Rocky's interface spec")


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

    success_rate = round(sum(1 for r in results if r["success"]) / len(results), 3)
    avg_tokens = round(sum(r["tokens_used"] for r in results) / len(results), 1)

    return {"success_rate": success_rate, "avg_tokens": avg_tokens}
