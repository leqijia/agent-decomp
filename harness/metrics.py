"""
Metrics for LLM agent failure decomposition experiments.

Functions:
    compute_alpha_decomposition -- failure attribution into context/env/capability
    compute_cohens_kappa        -- inter-annotator agreement
    compute_gamma_L             -- context gap by trajectory length bin
    compute_delta_synth         -- self-synthesis degradation gap by length bin

This module is the single source of truth for the proposal-defined length
bins (Section 2.4): <=10, 11-20, 21-40, 41-80. Trajectories outside that
range get None from `_length_bin` and are dropped from per-bin aggregates.
"""
from __future__ import annotations

import math
from collections import Counter


# ---------------------------------------------------------------------------
# Length-bin constants (single source of truth for the whole repo)
# ---------------------------------------------------------------------------

LENGTH_BINS: list[str] = ["<=10", "11-20", "21-40", "41-80"]

_LENGTH_BIN_ORDER: dict[str, int] = {b: i for i, b in enumerate(LENGTH_BINS)}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return (round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4))


def _diff_ci(
    full_results: list[dict],
    env_results: list[dict],
    z: float = 1.96,
) -> tuple[float, float]:
    """Normal-approximation CI for R_full - R_env.

    The context component is a signed difference, not a proportion. Prefer a
    paired interval when the two condition result lists are aligned; fall back
    to an independent two-proportion interval otherwise.
    """
    paired = [
        ((1 if f["success"] else 0) - (1 if e["success"] else 0))
        for f, e in zip(full_results, env_results)
        if f.get("success") is not None and e.get("success") is not None
    ]
    if len(paired) >= 2 and len(full_results) == len(env_results):
        mean = sum(paired) / len(paired)
        var = sum((x - mean) ** 2 for x in paired) / (len(paired) - 1)
        se = math.sqrt(var / len(paired))
    else:
        full_scored = [r for r in full_results if r.get("success") is not None]
        env_scored = [r for r in env_results if r.get("success") is not None]
        if not full_scored or not env_scored:
            return (0.0, 0.0)
        p_full = sum(1 for r in full_scored if r["success"]) / len(full_scored)
        p_env = sum(1 for r in env_scored if r["success"]) / len(env_scored)
        mean = p_full - p_env
        se = math.sqrt(
            p_full * (1 - p_full) / len(full_scored)
            + p_env * (1 - p_env) / len(env_scored)
        )

    return (
        round(max(-1.0, mean - z * se), 4),
        round(min(1.0, mean + z * se), 4),
    )


def _length_bin(total_steps: int) -> str | None:
    if total_steps <= 10:
        return "<=10"
    elif total_steps <= 20:
        return "11-20"
    elif total_steps <= 40:
        return "21-40"
    elif total_steps <= 80:
        return "41-80"
    return None  # outside defined bins


def _acc_and_count(results: list[dict]) -> tuple[float, int]:
    """Success rate and count from a list of result dicts (skip None success)."""
    scored = [r for r in results if r.get("success") is not None]
    if not scored:
        return 0.0, 0
    return round(sum(1 for r in scored if r["success"]) / len(scored), 4), len(scored)


def _bin_results(results: list[dict]) -> dict[str, list[dict]]:
    bins: dict[str, list[dict]] = {}
    for r in results:
        b = _length_bin(r.get("total_steps", 0))
        if b is not None:
            bins.setdefault(b, []).append(r)
    return bins


def _scored_by_task(results: list[dict]) -> dict[str, dict]:
    """Map task_id to scored result, dropping crash/unscored rows."""
    out: dict[str, dict] = {}
    for r in results:
        task_id = r.get("task_id")
        if task_id is None or r.get("success") is None:
            continue
        out[str(task_id)] = r
    return out


def _bool_acc(results: list[dict]) -> float | None:
    if not results:
        return None
    return round(sum(1 for r in results if r["success"]) / len(results), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_alpha_decomposition(full_results: list[dict], env_results: list[dict]) -> dict:
    """
    Decompose recovery rate into context, environment, and capability components.

    Args:
        full_results: run_episode results from the oracle (full) condition
        env_results:  run_episode results from the env_only condition

    Returns:
        {
            "R_full": float,
            "R_env": float,
            "alpha_context": float,
            "alpha_env": float,
            "alpha_capability": float,
            "ci_95": {
                "alpha_context": [lower, upper],
                "alpha_env": [lower, upper],
                "alpha_capability": [lower, upper],
            }
        }
    """
    R_full, n_full = _acc_and_count(full_results)
    R_env,  n_env  = _acc_and_count(env_results)

    alpha_context    = round(R_full - R_env, 4)
    alpha_env        = round(R_env, 4)
    alpha_capability = round(1.0 - R_full, 4)

    # Wilson CIs are valid for the proportion-valued components.
    succ_full = round(R_full * n_full)
    succ_env  = round(R_env  * n_env)

    # alpha_context is a signed difference of recovery rates; it can be
    # negative in partial runs or when oracle context hurts.
    ci_context    = _diff_ci(full_results, env_results)
    ci_env        = _wilson_ci(succ_env, n_env)
    ci_capability = _wilson_ci(n_full - succ_full, n_full)

    return {
        "R_full":            R_full,
        "R_env":             R_env,
        "alpha_context":     alpha_context,
        "alpha_env":         alpha_env,
        "alpha_capability":  alpha_capability,
        "ci_95": {
            "alpha_context":    list(ci_context),
            "alpha_env":        list(ci_env),
            "alpha_capability": list(ci_capability),
        },
    }


def compute_cohens_kappa(annotations_1: list[str], annotations_2: list[str]) -> float:
    """
    Compute Cohen's kappa inter-annotator agreement between two annotators.

    Args:
        annotations_1: list of string labels from annotator 1
        annotations_2: list of string labels from annotator 2

    Returns:
        Cohen's kappa as a float in [-1, 1], or 0.0 if undefined (all same label).

    Raises:
        ValueError if the two lists have different lengths.
    """
    if len(annotations_1) != len(annotations_2):
        raise ValueError(
            f"Annotation lists must have equal length, got "
            f"{len(annotations_1)} and {len(annotations_2)}"
        )

    n = len(annotations_1)
    if n == 0:
        return 0.0

    categories = sorted(set(annotations_1) | set(annotations_2))

    # observed agreement
    p_o = sum(a == b for a, b in zip(annotations_1, annotations_2)) / n

    # expected agreement
    count_1 = Counter(annotations_1)
    count_2 = Counter(annotations_2)
    p_e = sum((count_1[c] / n) * (count_2[c] / n) for c in categories)

    if p_e == 1.0:
        # all labels identical across both annotators — kappa undefined, return 0
        return 0.0

    return round((p_o - p_e) / (1.0 - p_e), 4)


def compute_gamma_L(oracle_results: list[dict], raw_results: list[dict]) -> dict:
    """
    Compute the context gap Gamma(L) per trajectory length bin.

    Gamma(L) = AccL(oracle) - AccL(raw)

    Args:
        oracle_results: run_episode results from the oracle condition
        raw_results:    run_episode results from the raw condition

    Returns:
        {
            "<=10":  {"oracle_acc": float, "raw_acc": float, "gamma": float, "count": int},
            "11-20": {...},
            "21-40": {...},
            "41-80": {...},
        }
        Only bins with at least one result in either condition are included.
    """
    oracle_by_task = _scored_by_task(oracle_results)
    raw_by_task = _scored_by_task(raw_results)
    common_task_ids = sorted(set(oracle_by_task) & set(raw_by_task))

    paired_bins: dict[str, tuple[list[dict], list[dict]]] = {}
    for task_id in common_task_ids:
        oracle = oracle_by_task[task_id]
        raw = raw_by_task[task_id]
        # Bin by the original/raw trajectory length so partial condition runs
        # do not shift examples across bins due to condition-specific stopping.
        b = _length_bin(raw.get("total_steps", 0))
        if b is None:
            continue
        oracle_bin, raw_bin = paired_bins.setdefault(b, ([], []))
        oracle_bin.append(oracle)
        raw_bin.append(raw)

    all_bins = sorted(paired_bins, key=lambda b: _LENGTH_BIN_ORDER[b])

    out = {}
    for b in all_bins:
        oracle_bin, raw_bin = paired_bins[b]
        oracle_acc = _bool_acc(oracle_bin)
        raw_acc = _bool_acc(raw_bin)
        count = min(len(oracle_bin), len(raw_bin))
        out[b] = {
            "oracle_acc": oracle_acc,
            "raw_acc":    raw_acc,
            "gamma":      (
                round(oracle_acc - raw_acc, 4)
                if oracle_acc is not None and raw_acc is not None
                else None
            ),
            "count":      count,
            "oracle_n":   len(oracle_bin),
            "raw_n":      len(raw_bin),
        }
    return out


def compute_delta_synth(oracle_results: list[dict], self_gen_results: list[dict]) -> dict:
    """
    Compute the self-synthesis degradation gap delta_synth per length bin.

    delta_synth(L) = AccL(oracle) - AccL(self_generated)

    Args:
        oracle_results:   run_episode results from the oracle condition
        self_gen_results: run_episode results from the self_generated condition

    Returns same structure as compute_gamma_L.
    """
    oracle_by_task = _scored_by_task(oracle_results)
    self_gen_by_task = _scored_by_task(self_gen_results)
    common_task_ids = sorted(set(oracle_by_task) & set(self_gen_by_task))

    paired_bins: dict[str, tuple[list[dict], list[dict]]] = {}
    for task_id in common_task_ids:
        oracle = oracle_by_task[task_id]
        self_gen = self_gen_by_task[task_id]
        b = _length_bin(oracle.get("total_steps", 0))
        if b is None:
            continue
        oracle_bin, self_gen_bin = paired_bins.setdefault(b, ([], []))
        oracle_bin.append(oracle)
        self_gen_bin.append(self_gen)

    all_bins = sorted(paired_bins, key=lambda b: _LENGTH_BIN_ORDER[b])

    out = {}
    for b in all_bins:
        oracle_bin, self_gen_bin = paired_bins[b]
        oracle_acc = _bool_acc(oracle_bin)
        self_gen_acc = _bool_acc(self_gen_bin)
        count = min(len(oracle_bin), len(self_gen_bin))
        out[b] = {
            "oracle_acc":   oracle_acc,
            "self_gen_acc": self_gen_acc,
            "delta_synth":  (
                round(oracle_acc - self_gen_acc, 4)
                if oracle_acc is not None and self_gen_acc is not None
                else None
            ),
            "count":        count,
            "oracle_n":     len(oracle_bin),
            "self_gen_n":   len(self_gen_bin),
        }
    return out
