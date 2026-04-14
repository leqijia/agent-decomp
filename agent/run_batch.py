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
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_CONFIG_DIR = Path(__file__).parent.parent / "config_files"
_OUT_DIR = Path(__file__).parent.parent / "trajectories" / "data"
_LOG_PATH = Path(__file__).parent.parent / "trajectories" / "batch_log.csv"


def _list_configs(config_dir: Path) -> list[Path]:
    return sorted(config_dir.glob("*.json"), key=lambda p: int(p.stem))


def _append_log(row: dict) -> None:
    write_header = not _LOG_PATH.exists()
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "task_id", "sites", "total_steps", "stop_reason",
            "success", "eval_score", "model",
        ])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument(
        "--thinking-budget", type=int, default=1024,
        help="Token budget for thinking mode (default: 1024)",
    )

    args = parser.parse_args()
    config_dir = Path(args.config_dir)
    out_dir = Path(args.out_dir)

    # Deferred import so --help is fast
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
                thinking_budget=args.thinking_budget,
            )
            result = run_episode(cfg, out_path=out_path)
            print(
                f"steps={result.total_steps} "
                f"stop={result.stop_reason} "
                f"score={result.eval_score}"
            )
            _append_log({
                "task_id": result.task_id,
                "sites": ",".join(result.sites),
                "total_steps": result.total_steps,
                "stop_reason": result.stop_reason,
                "success": result.success,
                "eval_score": result.eval_score,
                "model": result.model,
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
            })

    print("Batch complete.")


if __name__ == "__main__":
    main()
