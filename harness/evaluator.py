import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from llm.config import AGENT_MODEL

from baselines.acon import compress_trajectory


def _stub_llm(msg: str) -> str:
    """Stub LLM for ACON compression — returns a deterministic fake response."""
    return "### COMPRESSED_CONTEXT\n" + msg[:500]


def run_acon_compress(intent: str, steps: list[dict], guideline: str = "") -> dict:
    """
    Compress a trajectory using ACON with the stub LLM.

    This is the compression step that will be called inside run_episode()
    for condition='acon' once Rocky's episode interface is ready.

    Returns:
        {"compressed_text": str, "input_tokens": int, "model": str}
    """
    result = compress_trajectory(intent, steps, guideline, llm=_stub_llm)
    return {
        "compressed_text": result.compressed_text,
        "input_tokens": result.input_tokens,
        "model": result.model,
    }


def _length_bin(total_steps: int) -> str:
    if total_steps <= 10:
        return "<=10"
    elif total_steps <= 20:
        return "11-20"
    elif total_steps <= 40:
        return "21-40"
    elif total_steps <= 80:
        return "41-80"
    else:
        return ">80"


def run_episode(
    condition: str,
    config_file: str,
    *,
    model: str = AGENT_MODEL,
    max_steps: int = 30,
    max_obs_length: int = 1920,
    temperature: float = 0.0,
    window_size: int | None = None,
    oracle_regen_every_k: int | None = None,
    out_path: str | None = None,
) -> dict:
    """
    Run a single episode under a given experimental condition.

    Returns:
        dict with task_id, condition, success, tokens_used, eval_score, total_steps

    ACON condition (once Rocky's interface is ready):
        steps_so_far = rocky_interface.get_steps(config_file, up_to=t)
        intent = rocky_interface.get_intent(config_file)
        acon = run_acon_compress(intent, steps_so_far)
        # inject acon["compressed_text"] into agent context
        # tokens_used = acon["input_tokens"]
    """
    raise NotImplementedError("Pending Rocky's interface spec")


def run_intervention(
    trajectory_path: str,
    t_star: int,
    replacement_context: str,
    *,
    model: str = AGENT_MODEL,
    out_path: str | None = None,
) -> dict:
    """
    Re-run a trajectory from step t_star with a replacement context injected.

    Returns:
        dict with task_id, condition, success, tokens_used, eval_score, total_steps
    """
    raise NotImplementedError("Pending Rocky's interface spec")


def evaluate_batch(
    condition: str,
    config_files: list[str],
    *,
    out_dir: str,
    **kwargs,
) -> list[dict]:
    """
    Run run_episode on all config files for a given condition.

    Returns:
        list of run_episode result dicts
    """
    raise NotImplementedError("Pending Rocky's interface spec")


def compute_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics from a list of run_episode outputs.

    Args:
        results: list of dicts with keys: task_id, condition, success, tokens_used,
                 and optionally eval_score (float) and total_steps (int)

    Returns:
        {
            "success_rate": float,
            "avg_tokens": float,
            "avg_eval_score": float,
            "by_length_bin": {bin: {"success_rate": float, "avg_eval_score": float, "count": int}}
        }
    """
    if not results:
        return {
            "success_rate": 0.0,
            "avg_tokens": 0.0,
            "avg_eval_score": 0.0,
            "by_length_bin": {},
        }

    # overall success_rate — skip None
    scored = [r for r in results if r.get("success") is not None]
    success_rate = round(sum(1 for r in scored if r["success"]) / len(scored), 3) if scored else 0.0

    avg_tokens = round(sum(r["tokens_used"] for r in results) / len(results), 1)

    # avg_eval_score — skip None
    eval_scored = [r["eval_score"] for r in results if r.get("eval_score") is not None]
    avg_eval_score = round(sum(eval_scored) / len(eval_scored), 3) if eval_scored else 0.0

    # by_length_bin
    bins: dict[str, list[dict]] = {}
    for r in results:
        total_steps = r.get("total_steps")
        if total_steps is None:
            continue
        bin_key = _length_bin(total_steps)
        bins.setdefault(bin_key, []).append(r)

    by_length_bin = {}
    for bin_key, bin_results in bins.items():
        bin_scored = [r for r in bin_results if r.get("success") is not None]
        bin_success_rate = (
            round(sum(1 for r in bin_scored if r["success"]) / len(bin_scored), 3)
            if bin_scored else 0.0
        )
        bin_eval = [r["eval_score"] for r in bin_results if r.get("eval_score") is not None]
        bin_avg_eval = round(sum(bin_eval) / len(bin_eval), 3) if bin_eval else 0.0
        by_length_bin[bin_key] = {
            "success_rate": bin_success_rate,
            "avg_eval_score": bin_avg_eval,
            "count": len(bin_results),
        }

    return {
        "success_rate": success_rate,
        "avg_tokens": avg_tokens,
        "avg_eval_score": avg_eval_score,
        "by_length_bin": by_length_bin,
    }
