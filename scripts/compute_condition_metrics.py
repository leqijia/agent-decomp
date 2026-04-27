"""Compute aggregate metrics for any per-condition results.json.

Generic counterpart to `scripts/compute_exp3_raw_metrics.py`. Reads a
results.json produced by `scripts/build_condition_manifest.py` (or any
schema-compatible source) and emits a metrics.json with overall,
per-length-bin, per-site, and per-stop-reason success-rate breakdowns
(95% Wilson CIs), plus `avg_tokens` and `avg_eval_score` summaries that the
baselines table aggregator (`scripts/build_baselines_table.py`) consumes.

Usage:
    python scripts/compute_condition_metrics.py \\
        --results experiments/exp3/self_generated/results.json \\
        --out     experiments/exp3/self_generated/metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import LENGTH_BINS, _length_bin, _wilson_ci


def _scored(results: list[dict]) -> list[dict]:
    """Drop results where success is null (crash / evaluator unavailable)."""
    return [r for r in results if r.get("success") is not None]


def _success_rate_with_ci(results: list[dict]) -> dict:
    scored = _scored(results)
    n = len(scored)
    if n == 0:
        return {
            "n": 0,
            "n_unscored": len(results),
            "n_success": 0,
            "success_rate": 0.0,
            "ci_95": [0.0, 0.0],
        }
    n_success = sum(1 for r in scored if r["success"])
    rate = round(n_success / n, 4)
    lo, hi = _wilson_ci(n_success, n)
    return {
        "n": n,
        "n_unscored": len(results) - n,
        "n_success": n_success,
        "success_rate": rate,
        "ci_95": [lo, hi],
    }


def _site_label(sites: list | str | None) -> str:
    """Bucket multi-site results under 'multi_site' to keep the table compact."""
    if isinstance(sites, list):
        if len(sites) == 0:
            return "unknown"
        if len(sites) == 1:
            return str(sites[0])
        return "multi_site"
    if isinstance(sites, str):
        return sites or "unknown"
    return "unknown"


def _bin_results_proposal(results: list[dict]) -> dict[str, list[dict]]:
    """Bucket by harness.metrics._length_bin; out-of-range are bucketed separately."""
    bins: dict[str, list[dict]] = {}
    for r in results:
        b = _length_bin(r.get("total_steps", 0))
        if b is None:
            b = "out_of_range"
        bins.setdefault(b, []).append(r)
    return bins


def _step_token_total(step: dict) -> int:
    """prompt_tokens + completion_tokens for a single step (missing -> 0)."""
    return int(step.get("prompt_tokens") or 0) + int(step.get("completion_tokens") or 0)


def _task_token_total(result: dict) -> int:
    return sum(_step_token_total(s) for s in result.get("steps", []))


def _avg_tokens(results: list[dict]) -> dict:
    """Mean tokens-per-task across the whole condition.

    Tokens are summed across every step in every trajectory (regardless of
    success), giving the cost figure the baselines table reports as
    "Avg tokens / task". Reported alongside per-task min/max for sanity.
    """
    if not results:
        return {"n": 0, "avg_tokens": 0.0, "min_tokens": 0, "max_tokens": 0}
    per_task = [_task_token_total(r) for r in results]
    return {
        "n": len(per_task),
        "avg_tokens": round(sum(per_task) / len(per_task), 2),
        "min_tokens": min(per_task),
        "max_tokens": max(per_task),
    }


def _avg_eval_score(results: list[dict]) -> dict:
    """Mean of `eval_score` over scored tasks (eval_score not None)."""
    scored = [r for r in results if r.get("eval_score") is not None]
    if not scored:
        return {"n": 0, "avg_eval_score": None}
    vals = [float(r["eval_score"]) for r in scored]
    return {
        "n": len(vals),
        "avg_eval_score": round(sum(vals) / len(vals), 4),
        "min_eval_score": round(min(vals), 4),
        "max_eval_score": round(max(vals), 4),
    }


def compute_metrics(payload: dict) -> dict:
    results = payload.get("results", [])

    overall = _success_rate_with_ci(results)

    per_bin: dict[str, dict] = {}
    binned = _bin_results_proposal(results)
    for b in LENGTH_BINS:
        per_bin[b] = _success_rate_with_ci(binned.get(b, []))
    for b in sorted(set(binned.keys()) - set(LENGTH_BINS)):
        per_bin[b] = _success_rate_with_ci(binned[b])

    by_site: dict[str, list[dict]] = {}
    for r in results:
        by_site.setdefault(_site_label(r.get("sites")), []).append(r)
    per_site = {site: _success_rate_with_ci(rs) for site, rs in sorted(by_site.items())}

    stop_counts = dict(Counter(r.get("stop_reason", "unknown") for r in results))
    models = sorted({r.get("model", "") for r in results if r.get("model")})

    return {
        "condition":      payload.get("condition", "unknown"),
        "experiment_id":  payload.get("experiment_id", "unknown"),
        "model":          models[0] if len(models) == 1 else None,
        "models_present": models,
        "n_tasks":        len(results),
        "generated_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_results": payload.get("source_dir", ""),
        "overall":        overall,
        "tokens":         _avg_tokens(results),
        "eval_score":     _avg_eval_score(results),
        "by_length_bin":  per_bin,
        "by_site":        per_site,
        "stop_reasons":   stop_counts,
    }


def print_summary(metrics: dict) -> None:
    print(f"Condition:     {metrics['condition']}")
    print(f"Experiment:    {metrics['experiment_id']}")
    if metrics.get("model"):
        print(f"Model:         {metrics['model']}")
    elif metrics.get("models_present"):
        print(f"Models:        {metrics['models_present']}  (mixed-model suite)")
    print(f"Tasks scanned: {metrics['n_tasks']}")
    print()

    o = metrics["overall"]
    print(f"Overall success rate: {o['success_rate']:.4f}  "
          f"(95% CI {o['ci_95'][0]:.3f} - {o['ci_95'][1]:.3f})  "
          f"n={o['n']}  unscored={o['n_unscored']}")

    t = metrics["tokens"]
    if t["n"]:
        print(f"Avg tokens/task: {t['avg_tokens']:.1f}  "
              f"(min={t['min_tokens']}, max={t['max_tokens']})")

    es = metrics["eval_score"]
    if es["n"]:
        print(f"Avg eval_score:  {es['avg_eval_score']:.4f}  "
              f"(n={es['n']})")
    print()

    print("By length bin:")
    print(f"  {'bin':<14} {'n':>4} {'success':>9}  95% CI")
    for b, row in metrics["by_length_bin"].items():
        if row["n"] == 0 and row["n_unscored"] == 0:
            continue
        ci = row["ci_95"]
        print(f"  {b:<14} {row['n']:>4} {row['success_rate']:>9.4f}  "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    print()

    print("By site:")
    print(f"  {'site':<16} {'n':>4} {'success':>9}  95% CI")
    for site, row in metrics["by_site"].items():
        ci = row["ci_95"]
        print(f"  {site:<16} {row['n']:>4} {row['success_rate']:>9.4f}  "
              f"[{ci[0]:.3f}, {ci[1]:.3f}]")
    print()

    print("Stop reasons:")
    for reason, count in sorted(metrics["stop_reasons"].items(),
                                key=lambda x: (-x[1], x[0])):
        print(f"  {reason:<18} {count:>4}")


def run_compute(*, results_path: Path, out_path: Path) -> int:
    if not results_path.is_file():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        print("Run scripts/build_condition_manifest.py (or build_exp3_raw_manifest.py) first.",
              file=sys.stderr)
        return 1

    with open(results_path) as f:
        payload = json.load(f)

    metrics = compute_metrics(payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print_summary(metrics)
    print()
    print(f"Wrote metrics to {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-condition aggregate metrics from results.json."
    )
    parser.add_argument("--results", required=True,
                        help="Path to results.json from build_condition_manifest.py.")
    parser.add_argument("--out", required=True,
                        help="Output path for metrics.json.")
    args = parser.parse_args()

    return run_compute(results_path=Path(args.results), out_path=Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
