"""
Metrics for LLM agent failure decomposition experiments.

Functions:
    compute_alpha_decomposition -- failure attribution into context/env/capability
    compute_cohens_kappa        -- inter-annotator agreement
    compute_gamma_L             -- context gap by trajectory length bin
    compute_delta_synth         -- self-synthesis degradation gap by length bin
"""
from __future__ import annotations

import math
from collections import Counter


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

    # Wilson CIs treat each alpha as a proportion derived from the relevant sample
    succ_full = round(R_full * n_full)
    succ_env  = round(R_env  * n_env)

    # alpha_context: difference of two proportions — use the full-condition sample
    # as the anchor (conservative approximation)
    ci_context    = _wilson_ci(round(alpha_context * n_full), n_full)
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
    oracle_bins = _bin_results(oracle_results)
    raw_bins    = _bin_results(raw_results)
    all_bins    = sorted(set(oracle_bins) | set(raw_bins),
                         key=lambda b: {"<=10": 0, "11-20": 1, "21-40": 2, "41-80": 3}[b])

    out = {}
    for b in all_bins:
        oracle_acc, oracle_n = _acc_and_count(oracle_bins.get(b, []))
        raw_acc,    raw_n    = _acc_and_count(raw_bins.get(b, []))
        count = max(oracle_n, raw_n)
        out[b] = {
            "oracle_acc": oracle_acc,
            "raw_acc":    raw_acc,
            "gamma":      round(oracle_acc - raw_acc, 4),
            "count":      count,
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
    oracle_bins   = _bin_results(oracle_results)
    self_gen_bins = _bin_results(self_gen_results)
    all_bins = sorted(set(oracle_bins) | set(self_gen_bins),
                      key=lambda b: {"<=10": 0, "11-20": 1, "21-40": 2, "41-80": 3}[b])

    out = {}
    for b in all_bins:
        oracle_acc,   oracle_n   = _acc_and_count(oracle_bins.get(b, []))
        self_gen_acc, self_gen_n = _acc_and_count(self_gen_bins.get(b, []))
        count = max(oracle_n, self_gen_n)
        out[b] = {
            "oracle_acc":   oracle_acc,
            "self_gen_acc": self_gen_acc,
            "delta_synth":  round(oracle_acc - self_gen_acc, 4),
            "count":        count,
        }
    return out
