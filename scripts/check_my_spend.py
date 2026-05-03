"""Self-report your OpenRouter key usage + local cost-tracked spend.

Each teammate runs this once and pastes the output back, so we can
reconcile $5,850 of total account spend against per-key spend and
locally-tracked spend per machine.

Usage:
    python scripts/check_my_spend.py

Reads OPENROUTER_API_KEY from .env (or env var).
Calls https://openrouter.ai/api/v1/auth/key (shows YOUR key only —
no access to other keys on the account).

NOTHING is sent or modified. Read-only.
"""
import json
import os
import sys
import glob
import platform
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def probe_key(key):
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("data", {})
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def sum_local_costs():
    """Total cost recorded across local files on this machine."""
    totals = {}
    # Top-level ledgers
    for p in ["budget.json", "oracle/cost_log.json"]:
        full = REPO / p
        if full.exists():
            try:
                d = json.loads(full.read_text())
                totals[p] = d.get("total_cost_usd", 0) or 0
            except Exception:
                totals[p] = 0

    # Per-step costs in trajectory output dirs
    for label, pat in [
        ("trajectories/data/*",                          "trajectories/data/*.json"),
        ("trajectories/oracle_copies/*",                 "trajectories/oracle_copies/*.json"),
        ("trajectories/oracle_external/*",               "trajectories/oracle_external/*.json"),
        ("experiments/oracle_baseline/*",                "experiments/oracle_baseline/*.json"),
        ("experiments/exp3/oracle_external/data/*",      "experiments/exp3/oracle_external/data/*.json"),
        ("experiments/exp3/raw/data/*",                  "experiments/exp3/raw/data/*.json"),
        ("experiments/exp3/self_generated/data/*",       "experiments/exp3/self_generated/data/*.json"),
        ("experiments/sensitivity/*",                    "experiments/sensitivity/*.json"),
        ("experiments/ablations/*",                      "experiments/ablations/*.json"),
    ]:
        files = glob.glob(str(REPO / pat))
        if not files:
            continue
        n_total = n_zero = total = 0
        for fp in files:
            try:
                d = json.load(open(fp))
                c = sum((s.get("cost_usd") or 0) for s in (d.get("steps") or []))
                total += c
                n_total += 1
                if c == 0:
                    n_zero += 1
            except Exception:
                pass
        totals[label] = (total, n_total, n_zero)
    return totals


def main():
    print("=" * 60)
    print("OpenRouter spend self-report")
    print(f"machine:  {platform.node()}  ({platform.system()})")
    print(f"user:     {os.environ.get('USER') or os.environ.get('USERNAME')}")
    print(f"cwd:      {os.getcwd()}")
    print(f"time:     {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    key = load_key()
    if not key:
        print("\nNO API KEY FOUND in .env or env. Skipping key probe.\n")
    else:
        print(f"\n--- YOUR API KEY ---")
        print(f"prefix: {key[:10]}...{key[-4:]}  (length {len(key)})")
        info = probe_key(key)
        if "error" in info:
            print(f"probe error: {info['error']}")
        else:
            for k in ("label", "limit", "usage", "limit_remaining", "is_free_tier"):
                if k in info:
                    v = info[k]
                    if isinstance(v, float):
                        v = f"${v:.4f}"
                    print(f"  {k:18s} {v}")

    print(f"\n--- LOCAL TRACKED COSTS (on this machine, this clone) ---")
    t = sum_local_costs()
    flat = {k: v for k, v in t.items() if isinstance(v, (int, float))}
    nested = {k: v for k, v in t.items() if isinstance(v, tuple)}
    grand = sum(flat.values())
    for k, v in flat.items():
        print(f"  {k:50s}  ${v:>9.2f}")
        grand_add = 0
    for k, (cost, n_files, n_zero) in nested.items():
        flag = "  <-- token data missing!" if n_zero == n_files and n_files > 0 else ""
        print(f"  {k:50s}  ${cost:>9.2f}  ({n_files} files, {n_zero} with 0-cost){flag}")
        grand += cost
    print(f"  {'-'*50}  {'-'*9}")
    print(f"  {'GRAND TOTAL (locally tracked)':50s}  ${grand:>9.2f}")

    print(f"\n--- COMPARE ---")
    print(f"  If your KEY usage > local total by a lot, you ran something")
    print(f"  that didn't write cost data to disk (likely the agent loop")
    print(f"  for interventions/self-generated/oracle-external).")
    print(f"\n  Paste this entire output back to Rocky.")


if __name__ == "__main__":
    main()
