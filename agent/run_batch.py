"""Run a batch of WebArena tasks and collect trajectories.

Usage (on the VM):

    # Run specific task IDs
    python -m agent.run_batch --task-ids 0 5 12 156

    # Run a random sample of N tasks
    python -m agent.run_batch --sample 50

    # Run all eligible tasks
    python -m agent.run_batch --all

Trajectories are written to trajectories/data/<task_id>.json.
A summary CSV is appended to trajectories/batch_log.csv after each task.
Cumulative costs are tracked in budget.json at the repo root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "config_files"
_OUT_DIR = Path(__file__).parent.parent / "trajectories" / "data"
_LOG_PATH = Path(__file__).parent.parent / "trajectories" / "batch_log.csv"
_BUDGET_PATH = Path(__file__).parent.parent / "budget.json"


def _list_configs(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))


def _append_log(row: dict) -> None:
    write_header = not _LOG_PATH.exists()
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "sites", "total_steps", "stop_reason",
            "success", "eval_score", "model", "cost_usd",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _episode_cost(result) -> float:
    """Sum cost_usd from all steps in an episode result."""
    total = 0.0
    for step in result.steps:
        c = step.get("cost_usd")
        if c is not None:
            total += c
    return round(total, 6)


def _update_budget(episode_cost: float, model: str, task_id: int) -> dict:
    """Atomically read-modify-write budget.json with this episode's cost."""
    if _BUDGET_PATH.exists():
        with open(_BUDGET_PATH) as f:
            budget = json.load(f)
    else:
        budget = {
            "total_cost_usd": 0.0,
            "openrouter_budget_usd": 300.0,
            "by_model": {},
            "runs": [],
        }

    budget["total_cost_usd"] = round(budget["total_cost_usd"] + episode_cost, 6)
    if model not in budget["by_model"]:
        budget["by_model"][model] = {"cost_usd": 0.0, "episodes": 0}
    budget["by_model"][model]["cost_usd"] = round(
        budget["by_model"][model]["cost_usd"] + episode_cost, 6
    )
    budget["by_model"][model]["episodes"] += 1
    budget["runs"].append({
        "task_id": task_id,
        "model": model,
        "cost_usd": episode_cost,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    budget["remaining_usd"] = round(
        budget["openrouter_budget_usd"] - budget["total_cost_usd"], 6
    )

    with open(_BUDGET_PATH, "w") as f:
        json.dump(budget, f, indent=2)
    return budget


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run WebArena tasks.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-ids", nargs="+", type=int, help="Specific task IDs to run")
    group.add_argument("--sample", type=int, help="Random sample of N tasks")
    group.add_argument("--all", action="store_true", help="Run all eligible tasks")
    parser.add_argument("--config-dir", default=str(_CONFIG_DIR))
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--model", default=os.environ.get("AGENT_MODEL", "qwen/qwen3.5-27b"))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    parser.add_argument(
        "--thinking", action="store_true", default=True,
        help="Enable Qwen3 thinking mode (default: on)",
    )
    parser.add_argument(
        "--no-thinking", action="store_false", dest="thinking",
        help="Disable thinking mode",
    )

    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)

    from agent.react_agent import EpisodeConfig, run_episode

    all_configs = _list_configs(config_dir)
    if not all_configs:
        print(f"No config files found in {config_dir}. Run: python -m agent.prepare_configs")
        sys.exit(1)

    if args.task_ids:
        configs = [config_dir / f"{tid}.json" for tid in args.task_ids]
        missing = [p for p in configs if not p.exists()]
        if missing:
            print(f"Missing configs: {[str(m) for m in missing]}")
            sys.exit(1)
    elif args.sample:
        rng = random.Random(args.seed)
        configs = rng.sample(all_configs, min(args.sample, len(all_configs)))
    else:
        configs = all_configs

    print(f"Running {len(configs)} tasks with {args.model}, max_steps={args.max_steps}")
    print(f"Output: {out_dir}/")

    batch_cost = 0.0
    for i, config_path in enumerate(configs):
        task_id = int(config_path.stem)
        out_path = out_dir / f"{task_id}.json"

        if out_path.exists():
            print(f"[{i+1}/{len(configs)}] Skipping task {task_id} (already exists)")
            continue

        print(f"[{i+1}/{len(configs)}] Task {task_id} ... ", end="", flush=True)

        try:
            cfg = EpisodeConfig(
                config_file=str(config_path),
                model=args.model,
                max_steps=args.max_steps,
                thinking=args.thinking,
            )
            result = run_episode(cfg, out_path=out_path)
            ep_cost = _episode_cost(result)
            batch_cost += ep_cost
            budget = _update_budget(ep_cost, args.model, task_id)
            print(
                f"steps={result.total_steps} "
                f"stop={result.stop_reason} "
                f"score={result.eval_score} "
                f"cost=${ep_cost:.4f} "
                f"(total=${budget['total_cost_usd']:.4f}, "
                f"remaining=${budget['remaining_usd']:.2f})"
            )
            _append_log({
                "task_id": result.task_id,
                "sites": ",".join(result.sites),
                "total_steps": result.total_steps,
                "stop_reason": result.stop_reason,
                "success": result.success,
                "eval_score": result.eval_score,
                "model": result.model,
                "cost_usd": ep_cost,
            })
        except Exception:
            print(f"CRASH")
            traceback.print_exc()
            _append_log({
                "task_id": task_id,
                "sites": "",
                "total_steps": 0,
                "stop_reason": "crash",
                "success": None,
                "eval_score": None,
                "model": args.model,
                "cost_usd": 0.0,
            })

    print(f"\nBatch complete. Batch cost: ${batch_cost:.4f}")
    if _BUDGET_PATH.exists():
        with open(_BUDGET_PATH) as f:
            budget = json.load(f)
        print(f"Cumulative spend: ${budget['total_cost_usd']:.4f} / ${budget['openrouter_budget_usd']:.0f} "
              f"(${budget['remaining_usd']:.2f} remaining)")


if __name__ == "__main__":
    main()
