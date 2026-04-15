"""
API cost report for agent trajectory runs and oracle generation.

Usage:
    python scripts/calculate_costs.py
"""
import json
import os
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
TRAJ_DIR  = os.path.join(REPO_ROOT, "trajectories", "data")
COST_LOG  = os.path.join(REPO_ROOT, "oracle", "cost_log.json")

# $/1M tokens
MODEL_PRICING = {
    "qwen/qwen3.5-27b":               {"input": 0.14,  "output": 0.28},
    "google/gemini-2.5-flash":        {"input": 0.075, "output": 0.30},
    "google/gemini-3-flash-preview":  {"input": 0.075, "output": 0.30},
}
DEFAULT_PRICING = {"input": 0.14, "output": 0.28}

BUDGET_AGENT  = 1000.0
BUDGET_ORACLE =  300.0


def model_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def load_trajectories():
    by_model = defaultdict(lambda: {"trajs": 0, "input": 0, "output": 0, "cost": 0.0})
    total_trajs = total_in = total_out = 0

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
        in_tok  = sum(s.get("prompt_tokens",     0) for s in steps)
        out_tok = sum(s.get("completion_tokens",  0) for s in steps)
        cost = model_cost(model, in_tok, out_tok)

        by_model[model]["trajs"] += 1
        by_model[model]["input"] += in_tok
        by_model[model]["output"] += out_tok
        by_model[model]["cost"]  += cost

        total_trajs += 1
        total_in    += in_tok
        total_out   += out_tok

    total_cost = sum(v["cost"] for v in by_model.values())
    return total_trajs, total_in, total_out, total_cost, dict(by_model)


def load_oracle_costs():
    if not os.path.exists(COST_LOG):
        return {"total_calls": 0, "total_input_tokens": 0,
                "total_output_tokens": 0, "total_cost_usd": 0.0}
    with open(COST_LOG, encoding="utf-8") as f:
        return json.load(f)


def fmt(n: int) -> str:
    return f"{n:,}"


def main():
    total_trajs, total_in, total_out, agent_cost, by_model = load_trajectories()
    oracle = load_oracle_costs()
    oracle_cost = oracle.get("total_cost_usd", 0.0)
    grand_total = agent_cost + oracle_cost

    print("=== API Cost Report ===\n")

    print("Agent Trajectory Costs:")
    print(f"  Trajectories processed: {total_trajs}")
    print(f"  Total input tokens:     {fmt(total_in)}")
    print(f"  Total output tokens:    {fmt(total_out)}")
    if by_model:
        print("  Breakdown by model:")
        for model, d in sorted(by_model.items()):
            print(f"    {model}: ${d['cost']:.4f} ({d['trajs']} trajectories, "
                  f"{fmt(d['input'])} in / {fmt(d['output'])} out)")
    print(f"  Total agent cost:       ${agent_cost:.4f}")

    print()
    print("Oracle Generation Costs:")
    print(f"  Total calls:            {oracle.get('total_calls', 0)}")
    print(f"  Total input tokens:     {fmt(oracle.get('total_input_tokens', 0))}")
    print(f"  Total output tokens:    {fmt(oracle.get('total_output_tokens', 0))}")
    print(f"  Total oracle cost:      ${oracle_cost:.4f}")

    print()
    print(f"TOTAL SPEND SO FAR:       ${grand_total:.4f}")
    print("REMAINING BUDGET:")
    print(f"  OpenRouter (agent):  ${agent_cost:.2f} of ${BUDGET_AGENT:.0f} used, "
          f"${BUDGET_AGENT - agent_cost:.2f} remaining")
    print(f"  OpenRouter (oracle): ${oracle_cost:.2f} of ${BUDGET_ORACLE:.0f} used, "
          f"${BUDGET_ORACLE - oracle_cost:.2f} remaining")


if __name__ == "__main__":
    main()
