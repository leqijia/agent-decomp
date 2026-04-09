import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from harness.evaluator import compute_metrics, run_condition

# ---------------------------------------------------------------------------
# compute_metrics tests (existing)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# run_condition tests - sliding_window and observation_masking
# ---------------------------------------------------------------------------

fake_steps = [
    {"t": 1, "thought": "navigate", "action": "click('home')",
     "observation": "homepage loaded"},
    {"t": 2, "thought": "search", "action": "type('laptop')",
     "observation": "12 results found"},
    {"t": 3, "thought": "filter", "action": "click('price')",
     "observation": "sorted by price"},
]

# write fake trajectory to a temp file
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump({"steps": fake_steps}, f)
    tmp_path = f.name

try:
    sw_result = run_condition(tmp_path, "sliding_window", "fake_001", window_size=4000)
    print("\nsliding_window (max_tokens=4000):")
    print(f"  tokens_used:   {sw_result['tokens_used']}")
    print(f"  dropped_steps: {sw_result['dropped_steps']}")
    print(f"  success:       {sw_result['success']}")

    om_result = run_condition(tmp_path, "observation_masking", "fake_001")
    print("\nobservation_masking:")
    print(f"  tokens_used:   {om_result['tokens_used']}")
    print(f"  dropped_steps: {om_result['dropped_steps']}")
    print(f"  success:       {om_result['success']}")
finally:
    os.unlink(tmp_path)
