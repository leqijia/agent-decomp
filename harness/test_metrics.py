import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harness.metrics import (
    compute_alpha_decomposition,
    compute_cohens_kappa,
    compute_gamma_L,
    compute_delta_synth,
)


def _make_results(successes: list[bool], condition: str, total_steps_list: list[int] | None = None):
    return [
        {
            "task_id": f"task_{i:03d}",
            "condition": condition,
            "success": s,
            "tokens_used": 3000,
            "total_steps": (total_steps_list[i] if total_steps_list else 15),
        }
        for i, s in enumerate(successes)
    ]


# ---------------------------------------------------------------------------
# Test compute_alpha_decomposition
# ---------------------------------------------------------------------------
print("=== compute_alpha_decomposition ===")

# R_full = 0.6 (60/100), R_env = 0.2 (20/100)
full_results = _make_results([True] * 60 + [False] * 40, "oracle")
env_results  = _make_results([True] * 20 + [False] * 80, "env_only")

alpha = compute_alpha_decomposition(full_results, env_results)
print(json.dumps(alpha, indent=2))

assert alpha["R_full"] == 0.6,            f"R_full: {alpha['R_full']}"
assert alpha["R_env"]  == 0.2,            f"R_env: {alpha['R_env']}"
assert alpha["alpha_context"]    == 0.4,  f"alpha_context: {alpha['alpha_context']}"
assert alpha["alpha_env"]        == 0.2,  f"alpha_env: {alpha['alpha_env']}"
assert alpha["alpha_capability"] == 0.4,  f"alpha_capability: {alpha['alpha_capability']}"
assert alpha["alpha_context"] + alpha["alpha_env"] + alpha["alpha_capability"] == pytest_approx(1.0, abs=1e-3) \
    if False else True  # manual check below
total = round(alpha["alpha_context"] + alpha["alpha_env"] + alpha["alpha_capability"], 4)
assert total == 1.0, f"components should sum to 1.0, got {total}"
# CI bounds are valid ranges
for key, bounds in alpha["ci_95"].items():
    lower = -1.0 if key == "alpha_context" else 0.0
    assert lower <= bounds[0] <= bounds[1] <= 1.0, f"CI out of range for {key}: {bounds}"
negative_alpha = compute_alpha_decomposition(env_results, full_results)
print(json.dumps(negative_alpha, indent=2))
assert negative_alpha["alpha_context"] == -0.4
assert negative_alpha["ci_95"]["alpha_context"][0] < 0
print("PASS\n")


# ---------------------------------------------------------------------------
# Test compute_cohens_kappa
# ---------------------------------------------------------------------------
print("=== compute_cohens_kappa ===")

annotator_1 = ["context", "context", "capability", "context", "capability"]
annotator_2 = ["context", "capability", "capability", "context", "context"]

kappa = compute_cohens_kappa(annotator_1, annotator_2)
print(f"kappa = {kappa}")
assert -1.0 <= kappa <= 1.0, f"kappa out of range: {kappa}"
print("PASS")

# Edge case: perfect agreement
kappa_perfect = compute_cohens_kappa(["a", "b", "a"], ["a", "b", "a"])
print(f"kappa (perfect agreement) = {kappa_perfect}")
assert kappa_perfect == 1.0, f"expected 1.0, got {kappa_perfect}"
print("PASS")

# Edge case: all same label
kappa_uniform = compute_cohens_kappa(["context"] * 5, ["context"] * 5)
print(f"kappa (all same label, undefined) = {kappa_uniform}")
assert kappa_uniform == 0.0, f"expected 0.0, got {kappa_uniform}"
print("PASS\n")


# ---------------------------------------------------------------------------
# Test compute_gamma_L
# ---------------------------------------------------------------------------
print("=== compute_gamma_L ===")

# Distribute tasks across all 4 bins
steps_by_bin = [5, 5, 8, 15, 15, 25, 25, 35, 50, 70]

oracle_results = _make_results(
    [True, True, True, True, False, True, False, True, False, True],
    "oracle", steps_by_bin
)
raw_results = _make_results(
    [True, False, False, True, False, False, False, False, False, False],
    "raw", steps_by_bin
)

gamma = compute_gamma_L(oracle_results, raw_results)
print(json.dumps(gamma, indent=2))

assert set(gamma.keys()) == {"<=10", "11-20", "21-40", "41-80"}, \
    f"unexpected bins: {set(gamma.keys())}"
for b, vals in gamma.items():
    assert "oracle_acc" in vals and "raw_acc" in vals and "gamma" in vals and "count" in vals
    assert vals["gamma"] == round(vals["oracle_acc"] - vals["raw_acc"], 4), \
        f"gamma mismatch in bin {b}"

empty_gamma = compute_gamma_L(
    [{"task_id": "task_000", "success": None, "total_steps": 5}],
    [{"task_id": "task_000", "success": True, "total_steps": 5}],
)
assert empty_gamma == {}, f"unscored oracle rows should not produce Gamma: {empty_gamma}"
print("PASS\n")


# ---------------------------------------------------------------------------
# Test compute_delta_synth
# ---------------------------------------------------------------------------
print("=== compute_delta_synth ===")

self_gen_results = _make_results(
    [True, False, False, True, False, False, False, False, False, False],
    "self_generated", steps_by_bin
)

delta = compute_delta_synth(oracle_results, self_gen_results)
print(json.dumps(delta, indent=2))

assert set(delta.keys()) == {"<=10", "11-20", "21-40", "41-80"}
for b, vals in delta.items():
    assert "oracle_acc" in vals and "self_gen_acc" in vals and "delta_synth" in vals
    assert vals["delta_synth"] == round(vals["oracle_acc"] - vals["self_gen_acc"], 4)

empty_delta = compute_delta_synth(
    [{"task_id": "task_000", "success": None, "total_steps": 5}],
    [{"task_id": "task_000", "success": True, "total_steps": 5}],
)
assert empty_delta == {}, f"unscored oracle rows should not produce delta_synth: {empty_delta}"
print("PASS\n")

print("All tests passed.")
