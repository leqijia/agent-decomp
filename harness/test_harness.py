import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harness.evaluator import compute_metrics

# ---------------------------------------------------------------------------
# compute_metrics — legacy tests (success only, no eval_score/total_steps)
# ---------------------------------------------------------------------------

fake_results_1 = [
    {"task_id": "task_001", "condition": "sliding_window", "success": True, "tokens_used": 3200},
    {"task_id": "task_002", "condition": "sliding_window", "success": False, "tokens_used": 4100},
    {"task_id": "task_003", "condition": "sliding_window", "success": True, "tokens_used": 2900},
]
fake_results_2 = [
    {"task_id": "task_004", "condition": "raw", "success": False, "tokens_used": 5000},
    {"task_id": "task_005", "condition": "raw", "success": False, "tokens_used": 4800},
]
fake_results_3 = []

print("Test 1 (mixed):", compute_metrics(fake_results_1))
print("Test 2 (all failed):", compute_metrics(fake_results_2))
print("Test 3 (empty):", compute_metrics(fake_results_3))

# ---------------------------------------------------------------------------
# compute_metrics — new fields: eval_score, total_steps, by_length_bin
# ---------------------------------------------------------------------------

fake_results = [
    {"task_id": "task_001", "condition": "raw", "success": True,
     "tokens_used": 3200, "eval_score": 1.0, "total_steps": 8},
    {"task_id": "task_002", "condition": "raw", "success": False,
     "tokens_used": 4100, "eval_score": 0.0, "total_steps": 25},
    {"task_id": "task_003", "condition": "raw", "success": True,
     "tokens_used": 2900, "eval_score": 0.8, "total_steps": 15},
    {"task_id": "task_004", "condition": "sliding_window", "success": False,
     "tokens_used": 3800, "eval_score": 0.2, "total_steps": 45},
]

metrics = compute_metrics(fake_results)
print("\nTest 4 (eval_score + total_steps):")
print(json.dumps(metrics, indent=2))

# verify by_length_bin grouping
# task_001: total_steps=8  -> "<=10"
# task_003: total_steps=15 -> "11-20"
# task_002: total_steps=25 -> "21-40"
# task_004: total_steps=45 -> "41-80"
expected_bins = {"<=10", "11-20", "21-40", "41-80"}
actual_bins = set(metrics["by_length_bin"].keys())
assert actual_bins == expected_bins, f"Expected bins {expected_bins}, got {actual_bins}"
assert metrics["by_length_bin"]["<=10"]["count"] == 1
assert metrics["by_length_bin"]["11-20"]["count"] == 1
assert metrics["by_length_bin"]["21-40"]["count"] == 1
assert metrics["by_length_bin"]["41-80"]["count"] == 1
print("\nby_length_bin grouping: OK")
