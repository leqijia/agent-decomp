import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harness.evaluator import compute_metrics

# Test 1 - mixed results
fake_results_1 = [
    {"task_id": "task_001", "condition": "sliding_window", "success": True, "tokens_used": 3200},
    {"task_id": "task_002", "condition": "sliding_window", "success": False, "tokens_used": 4100},
    {"task_id": "task_003", "condition": "sliding_window", "success": True, "tokens_used": 2900},
]

# Test 2 - all failed
fake_results_2 = [
    {"task_id": "task_004", "condition": "raw", "success": False, "tokens_used": 5000},
    {"task_id": "task_005", "condition": "raw", "success": False, "tokens_used": 4800},
]

# Test 3 - empty results (edge case)
fake_results_3 = []

print("Test 1 (mixed):", compute_metrics(fake_results_1))
print("Test 2 (all failed):", compute_metrics(fake_results_2))
print("Test 3 (empty):", compute_metrics(fake_results_3))
