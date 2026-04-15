"""
API cost report for agent trajectory runs and oracle generation.

Two cost sources:
  1. budget.json (authoritative) — actual costs from OpenRouter usage.cost
     field, written by run_batch.py after each episode.
  2. Trajectory files (fallback) — estimates cost from token counts and
     per-model pricing when budget.json is missing or for old trajectories
     that predate the cost_usd field.

Usage:
    python scripts/calculate_costs.py
"""
import json
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
TRAJ_DIR  = os.path.join(REPO_ROOT, "trajectories", "data")
COST_LOG  = os.path.join(REPO_ROOT, "oracle", "cost_log.json")
BUDGET_PATH = os.path.join(REPO_ROOT, "budget.json")

# $/1M tokens — fallback when cost_usd is not in trajectory steps
MODEL_PRICING = {
    "qwen/qwen3.5-27b":               {"input": 0.14,  "output": 0.28},
    "google/gemini-2.5-flash":        {"input": 0.075, "output": 0.30},
    "google/gemini-3-flash-preview":  {"input": 0.075, "output": 0.30},
}
DEFAULT_PRICING = {"input": 0.14, "output": 0.28}

BUDGET_OPENROUTER = 300.0


def model_cost_estimate(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def load_trajectories():
    by_model = defaultdict(lambda: {
        "trajs": 0, "input": 0, "output": 0,
        "cost_reported": 0.0, "cost_estimated": 0.0,
    })
    total_trajs = total_in = total_out = 0
    total_reported = 0.0
    total_estimated = 0.0

    for fname in sorted(os.listdir(TRAJ_DIR)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(TRAJ_DIR, fname)
        try:
            with open(fpath, encoding='utf-8', errors='ignore') as f:
                traj = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        steps = [s for s in traj.get("steps", []) if "t" in s]
        if not steps:
            continue

        model = traj.get("model", "unknown")
        in_tok  = sum(s.get("prompt_tokens") or 0 for s in steps)
        out_tok = sum(s.get("completion_tokens") or 0 for s in steps)

        reported = sum(s.get("cost_usd") or 0.0 for s in steps)
        estimated = model_cost_estimate(model, in_tok, out_tok)

        by_model[model]["trajs"] += 1
        by_model[model]["input"] += in_tok
        by_model[model]["output"] += out_tok
        by_model[model]["cost_reported"] += reported
        by_model[model]["cost_estimated"] += estimated

        total_trajs += 1
        total_in    += in_tok
        total_out   += out_tok
        total_reported += reported
        total_estimated += estimated

    return total_trajs, total_in, total_out, total_reported, total_estimated, dict(by_model)


def load_oracle_costs():
    if not os.path.exists(COST_LOG):
        return {"total_calls": 0, "total_input_tokens": 0,
                "total_output_tokens": 0, "total_cost_usd": 0.0}
    with open(COST_LOG, encoding="utf-8") as f:
        return json.load(f)


def load_budget():
    if not os.path.exists(BUDGET_PATH):
        return None
    with open(BUDGET_PATH) as f:
        return json.load(f)


def fmt(n: int) -> str:
    return f"{n:,}"


def main():
    total_trajs, total_in, total_out, reported, estimated, by_model = load_trajectories()
    oracle = load_oracle_costs()
    oracle_cost = oracle.get("total_cost_usd", 0.0)
    budget = load_budget()

    agent_cost = reported if reported > 0 else estimated
    cost_source = "OpenRouter reported" if reported > 0 else "estimated from tokens"

    grand_total = agent_cost + oracle_cost

    print("=== API Cost Report ===\n")

    if budget:
        print("--- budget.json (authoritative) ---")
        print(f"  Total spend:   ${budget['total_cost_usd']:.4f}")
        print(f"  Budget:        ${budget['openrouter_budget_usd']:.0f}")
        print(f"  Remaining:     ${budget['remaining_usd']:.2f}")
        if budget.get("by_model"):
            print("  By model:")
            for m, d in sorted(budget["by_model"].items()):
                print(f"    {m}: ${d['cost_usd']:.4f} ({d['episodes']} episodes)")
        print()

    print(f"Agent Trajectory Costs ({cost_source}):")
    print(f"  Trajectories processed: {total_trajs}")
    print(f"  Total input tokens:     {fmt(total_in)}")
    print(f"  Total output tokens:    {fmt(total_out)}")
    if by_model:
        print("  Breakdown by model:")
        for model, d in sorted(by_model.items()):
            cost_val = d['cost_reported'] if d['cost_reported'] > 0 else d['cost_estimated']
            print(f"    {model}: ${cost_val:.4f} ({d['trajs']} trajectories, "
                  f"{fmt(d['input'])} in / {fmt(d['output'])} out)")
    print(f"  Total agent cost:       ${agent_cost:.4f}")

    print()
    print("Oracle Generation Costs:")
    print(f"  Total calls:            {oracle.get('total_calls', 0)}")
    print(f"  Total input tokens:     {fmt(oracle.get('total_input_tokens', 0))}")
    print(f"  Total output tokens:    {fmt(oracle.get('total_output_tokens', 0))}")
    print(f"  Total oracle cost:      ${oracle_cost:.4f}")

    print()
    print(f"GRAND TOTAL SPEND:        ${grand_total:.4f}")
    print(f"REMAINING BUDGET:         ${BUDGET_OPENROUTER - grand_total:.2f} of ${BUDGET_OPENROUTER:.0f}")


if __name__ == "__main__":
    main()
