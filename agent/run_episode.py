"""CLI entry point: run a single WebArena task and emit a trajectory JSON.

Usage (from repo root, on the VM where WebArena is running):

    python -m agent.run_episode \
        --config-file config_files/156.json \
        --out-dir trajectories/data/

Reads AGENT_MODEL from .env. Override with --model if needed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one WebArena task with the ReAct agent."
    )
    parser.add_argument(
        "--config-file", required=True,
        help="Path to the WebArena task JSON (e.g. config_files/156.json)",
    )
    parser.add_argument(
        "--out-dir", default="trajectories/data",
        help="Directory for output JSON. File is named <task_id>.json.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("AGENT_MODEL", "qwen/qwen3.5-27b"),
        help="Model slug — OpenRouter (default: $AGENT_MODEL from .env)",
    )
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-obs-length", type=int, default=1920)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--headless", action="store_true", default=True,
        help="Run browser headless (default: True)",
    )
    parser.add_argument(
        "--no-headless", action="store_false", dest="headless",
        help="Show the browser window (for debugging)",
    )

    args = parser.parse_args()

    # Deferred import so --help is fast and doesn't require playwright
    from agent.react_agent import EpisodeConfig, run_episode

    with open(args.config_file) as f:
        task_id = json.load(f)["task_id"]

    out_path = Path(args.out_dir) / f"{task_id}.json"

    cfg = EpisodeConfig(
        config_file=args.config_file,
        model=args.model,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_obs_length=args.max_obs_length,
        headless=args.headless,
    )

    print(f"Running task {task_id} with {cfg.model}, max_steps={cfg.max_steps}")
    result = run_episode(cfg, out_path=out_path)
    print(
        f"Done: task={result.task_id} "
        f"steps={result.total_steps} "
        f"stop={result.stop_reason} "
        f"score={result.eval_score} "
        f"-> {out_path}"
    )


if __name__ == "__main__":
    main()
