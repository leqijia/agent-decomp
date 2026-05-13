"""Generic batch driver for any experimental condition.

Wraps `harness.evaluator.evaluate_batch` so a single CLI can run the raw,
self-generated, oracle_external, env_only, and the five baselines
(sliding_window, observation_masking, acon, agentdiet, perfect_retrieval)
on the same task suite that backs the Experiment 3 raw condition.

Default task suite is the existing raw manifest (`experiments/exp3/raw/manifest.csv`),
which is the suite contract the proposal pins for cross-condition comparison.

Usage:
    python scripts/run_condition_batch.py --condition self_generated
    python scripts/run_condition_batch.py --condition acon \
        --out-dir experiments/baselines/acon/data
    python scripts/run_condition_batch.py --condition sliding_window \
        --task-ids 28 49 102
    python scripts/run_condition_batch.py --condition perfect_retrieval \
        --task-list experiments/exp3/raw/manifest.csv \
        --model qwen/qwen3.5-27b
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from harness.conditions import VALID_CONDITIONS
from harness.evaluator import evaluate_batch
from llm.config import AGENT_MODEL

load_dotenv()


_DEFAULT_TASK_LIST = "experiments/exp3/raw/manifest.csv"
_DEFAULT_CONFIG_DIR = "config_files"


def _default_out_dir(condition: str) -> str:
    """Per-condition default output directory.

    Mirrors the layout the analysis pipeline expects:
        experiments/exp3/raw/data/                  (fresh raw runs)
        experiments/exp3/self_generated/data/
        experiments/baselines/<method>/data/        (the five baselines)
        experiments/exp3/<other>/data/              (env_only, oracle_external, ...)
    """
    if condition == "raw":
        return "experiments/exp3/raw/data"
    if condition in {"self_generated", "oracle_external", "env_only"}:
        return f"experiments/exp3/{condition}/data"
    return f"experiments/baselines/{condition}/data"


def _load_task_ids(task_list: str | None, task_ids_arg: list[int] | None) -> list[int]:
    """Resolve a list of integer task ids from CLI args.

    Precedence: explicit --task-ids > --task-list (CSV with task_id column or
    JSON list/dict).
    """
    if task_ids_arg:
        return list(task_ids_arg)

    path = Path(task_list or _DEFAULT_TASK_LIST)
    if not path.is_file():
        raise FileNotFoundError(
            f"Task list not found: {path}. Pass --task-list or --task-ids."
        )

    if path.suffix.lower() == ".csv":
        ids: list[int] = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "task_id" not in reader.fieldnames:
                raise ValueError(
                    f"{path} has no 'task_id' column (got: {reader.fieldnames})."
                )
            for row in reader:
                raw = (row.get("task_id") or "").strip()
                if not raw:
                    continue
                try:
                    ids.append(int(raw))
                except ValueError:
                    print(f"  WARNING: skipping non-integer task_id {raw!r}", file=sys.stderr)
        return ids

    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [int(x) for x in payload]
        if isinstance(payload, dict):
            return [int(k) for k in payload.keys()]
        raise ValueError(f"{path} JSON must be a list or dict; got {type(payload).__name__}")

    raise ValueError(f"Unsupported task list format: {path.suffix}")


def _resolve_config_files(task_ids: list[int], config_dir: Path) -> list[str]:
    """Map task_ids to config_files/<task_id>.json paths; abort on any missing."""
    paths: list[str] = []
    missing: list[int] = []
    for tid in task_ids:
        p = config_dir / f"{tid}.json"
        if not p.exists():
            missing.append(tid)
        else:
            paths.append(str(p))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} config files missing under {config_dir}: "
            f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
        )
    return paths


def _refresh_auth_cookies() -> None:
    """Same WebArena auth refresh that agent/run_batch.py performs before a batch."""
    print("Refreshing auth cookies ...", end=" ", flush=True)
    try:
        from webarena.browser_env.auto_login import main as auto_login_main
        auto_login_main()
        print("done.")
    except Exception as e:
        print(f"WARNING: auto_login failed ({e}). Cookies may be stale.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of WebArena tasks under a given experimental condition."
    )
    parser.add_argument(
        "--condition", required=True, choices=sorted(VALID_CONDITIONS),
        help="Which experimental condition to run (see harness.conditions.VALID_CONDITIONS).",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--task-list", default=None,
        help=f"CSV (with task_id column) or JSON list/dict of task ids "
             f"(default: {_DEFAULT_TASK_LIST})",
    )
    src.add_argument(
        "--task-ids", nargs="+", type=int,
        help="Specific task ids to run (overrides --task-list).",
    )
    parser.add_argument(
        "--config-dir", default=_DEFAULT_CONFIG_DIR,
        help=f"Directory of WebArena task configs (default: {_DEFAULT_CONFIG_DIR})",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for trajectory JSONs "
             "(default: per-condition; see _default_out_dir).",
    )
    parser.add_argument("--model", default=AGENT_MODEL,
                        help=f"Agent model slug (default: {AGENT_MODEL})")
    parser.add_argument("--max-steps", type=int, default=75)
    parser.add_argument("--max-obs-length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--thinking", action="store_true", default=True,
        help="Enable Qwen3 thinking mode (default: on)",
    )
    parser.add_argument(
        "--no-thinking", action="store_false", dest="thinking",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--skip-auto-login", action="store_true",
        help="Skip the WebArena cookie refresh (useful for dry-run / local tests).",
    )
    parser.add_argument(
        "--rerun-crashes", action="store_true",
        help="Rerun existing retryable crash outputs instead of treating them as complete.",
    )
    parser.add_argument(
        "--perfect-retrieval-k", type=int, default=None,
        help="k for the perfect_retrieval policy (only used when --condition=perfect_retrieval).",
    )
    parser.add_argument(
        "--window-size", type=int, default=None,
        help="Token window for sliding_window (default: harness fallback 4096).",
    )

    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f"ERROR: config dir not found: {config_dir}", file=sys.stderr)
        return 1

    try:
        task_ids = _load_task_ids(args.task_list, args.task_ids)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not task_ids:
        print("ERROR: empty task list.", file=sys.stderr)
        return 1

    try:
        config_files = _resolve_config_files(task_ids, config_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or _default_out_dir(args.condition)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Condition:   {args.condition}")
    print(f"Model:       {args.model}")
    print(f"Tasks:       {len(config_files)}  (from "
          f"{'--task-ids' if args.task_ids else (args.task_list or _DEFAULT_TASK_LIST)})")
    print(f"Out dir:     {out_dir}")
    print(f"Max steps:   {args.max_steps}")
    print(f"Thinking:    {args.thinking}")

    if not args.skip_auto_login:
        _refresh_auth_cookies()

    kwargs: dict = {
        "model": args.model,
        "max_steps": args.max_steps,
        "max_obs_length": args.max_obs_length,
        "temperature": args.temperature,
        "thinking": args.thinking,
        "rerun_crashes": args.rerun_crashes,
    }
    if args.window_size is not None:
        kwargs["window_size"] = args.window_size
    if args.perfect_retrieval_k is not None:
        kwargs["perfect_retrieval_k"] = args.perfect_retrieval_k

    results = evaluate_batch(args.condition, config_files, out_dir=out_dir, **kwargs)

    n = len(results)
    n_success = sum(1 for r in results if r.get("success") is True)
    n_failure = sum(1 for r in results if r.get("success") is False)
    n_unscored = n - n_success - n_failure

    print()
    print(f"Done. {n} episodes under condition={args.condition!r}")
    print(f"  success:  {n_success}")
    print(f"  failure:  {n_failure}")
    print(f"  unscored: {n_unscored}  (success is null -- crash / evaluator error)")
    print(f"Trajectories written to {out_dir}/")
    print()
    print("Next:")
    print(f"  python scripts/build_condition_manifest.py "
          f"--data-dir {out_dir} "
          f"--out-dir {os.path.dirname(out_dir.rstrip('/'))} "
          f"--condition {args.condition} "
          f"--experiment-id <exp_id>")
    print(f"  python scripts/compute_condition_metrics.py "
          f"--results {os.path.dirname(out_dir.rstrip('/'))}/results.json "
          f"--out {os.path.dirname(out_dir.rstrip('/'))}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
