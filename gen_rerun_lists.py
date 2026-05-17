import json, random
from collections import defaultdict

random.seed(42)

raw = json.load(open("experiments/exp3/raw/results.json"))["results"]

def length_bin(n):
    if n <= 10: return "<=10"
    if n <= 20: return "11-20"
    if n <= 40: return "21-40"
    if n <= 80: return "41-80"
    return None

# Build raw scored trajectories by bin
by_bin = defaultdict(list)
for r in raw:
    if r.get("success") is None:
        continue
    b = length_bin(r.get("total_steps", 0))
    if b:
        by_bin[b].append(r["task_id"])

print("=" * 70)
print("LONG-TAIL k=3 RERUN (already sent to Adithya)")
print("=" * 70)
print()
print("Run k=3 oracle_external on these trajectories to fill in")
print("the 21-40 and 41-80 bins for the headline Gamma(L) figure.")
print()

# Long-tail list (already sent)
import os
done = set()
for f in os.listdir("trajectories/oracle_external"):
    if f.endswith(".json"):
        try:
            done.add(int(f.split(".")[0]))
        except ValueError:
            pass

long_21_40 = [t for t in by_bin["21-40"] if t not in done]
long_41_80 = [t for t in by_bin["41-80"] if t not in done]

print(f"21-40 bin ({len(long_21_40)} trajectories):")
print(f"  {sorted(long_21_40)}")
print()
print(f"41-80 bin ({len(long_41_80)} trajectories):")
print(f"  {sorted(long_41_80)}")
print()
print(f"Total k=3 long-tail rerun: {len(long_21_40) + len(long_41_80)} trajectories")
print()
print()

print("=" * 70)
print("k=1 AND k=5 SENSITIVITY SUBSET (use SAME list for both k values)")
print("=" * 70)
print()
print("Stratified 10 per bin = 40 trajectories. Same task IDs for")
print("k=1 AND k=5 so within-trajectory comparison is meaningful.")
print()

target_per_bin = 10
sensitivity_subset = []
for b in ["<=10", "11-20", "21-40", "41-80"]:
    avail = by_bin[b]
    if len(avail) <= target_per_bin:
        chosen = avail
    else:
        chosen = random.sample(avail, target_per_bin)
    sensitivity_subset.extend(chosen)
    print(f"{b}: {sorted(chosen)} (n={len(chosen)})")

print()
print(f"Total sensitivity subset: {len(sensitivity_subset)} trajectories")
print(f"Run k=1 on these {len(sensitivity_subset)} tasks")
print(f"Run k=5 on the SAME {len(sensitivity_subset)} tasks")
print()
print("Full sensitivity task list (same for k=1 and k=5):")
print(sorted(sensitivity_subset))
