"""Shared helpers for per-condition manifest building.

Used by both:
- `scripts/build_exp3_raw_manifest.py` (raw-condition sidecar manifest)
- `scripts/build_condition_manifest.py` (any other condition)

Single source of truth for manifest CSV columns, results JSON shape, and the
pipeline-anomaly predicate, so that every per-condition manifest in
`experiments/.../*/manifest.csv` is schema-compatible with downstream
analysis (`scripts/compute_condition_metrics.py`,
`scripts/build_baselines_table.py`, `harness.metrics.compute_*`).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.metrics import _length_bin


PIPELINE_ANOMALY_REASONS = {"parse_failures", "crash", "replay_desync"}

REQUIRED_FIELDS = ("task_id", "total_steps", "stop_reason")

MANIFEST_COLUMNS = [
    "task_id",
    "trajectory_path",
    "sites",
    "model",
    "total_steps",
    "length_bin",
    "stop_reason",
    "success",
    "eval_score",
    "condition",
    "experiment_id",
]

SKIPPED_COLUMNS = [
    "task_id",
    "trajectory_path",
    "stop_reason",
    "reason",
]


def _load_trajectory(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: could not load {path.name}: {e}", file=sys.stderr)
        return None


def _is_conformant(data: dict, path: Path) -> bool:
    """Light sanity check: required top-level fields present.

    The full schema lives in `trajectories/SPEC.md` and is enforced by the
    (TBD) `trajectories/validate.py` validator. Here we only need enough
    fields to populate the manifest and downstream metrics.
    """
    for field in REQUIRED_FIELDS:
        if field not in data:
            print(
                f"  WARNING: {path.name} missing required field {field!r}, skipping",
                file=sys.stderr,
            )
            return False
    return True


def _format_sites(sites) -> str:
    """Match the comma-joined format used by trajectories/batch_log.csv."""
    if isinstance(sites, list):
        return ",".join(str(s) for s in sites)
    if isinstance(sites, str):
        return sites
    return ""


def _resolve_traj_path(path: Path) -> str:
    """Repo-relative path when possible; absolute fallback for cross-drive cases."""
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path.resolve())


def build_manifest_rows(
    data_dir: Path, condition: str, experiment_id: str
) -> tuple[list[dict], list[dict], list[dict]]:
    """Scan data_dir and return (manifest_rows, result_dicts, skipped_rows).

    `manifest_rows` is the per-task table (CSV form). `result_dicts` is the
    list-of-dicts shape that `harness.metrics.compute_alpha_decomposition` /
    `compute_gamma_L` consume. `skipped_rows` flags pipeline anomalies; those
    rows ARE still kept in the main artifacts.
    """
    manifest_rows: list[dict] = []
    result_dicts: list[dict] = []
    skipped_rows: list[dict] = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        path = data_dir / fname
        data = _load_trajectory(path)
        if data is None or not _is_conformant(data, path):
            continue

        total_steps = data.get("total_steps", 0)
        stop_reason = data.get("stop_reason", "unknown")
        sites_raw = data.get("sites", [])
        success = data.get("success")
        eval_score = data.get("eval_score")
        model = data.get("model", "")
        intent = data.get("intent", "")
        agent_variant = data.get("agent_variant", "")

        try:
            task_id = int(data["task_id"])
        except (TypeError, ValueError):
            task_id = data["task_id"]

        bin_label = _length_bin(total_steps)
        if bin_label is None:
            bin_label = "out_of_range"

        traj_path = _resolve_traj_path(path)

        manifest_rows.append({
            "task_id": task_id,
            "trajectory_path": traj_path,
            "sites": _format_sites(sites_raw),
            "model": model,
            "total_steps": total_steps,
            "length_bin": bin_label,
            "stop_reason": stop_reason,
            "success": success,
            "eval_score": eval_score,
            "condition": condition,
            "experiment_id": experiment_id,
        })

        result_dicts.append({
            "task_id": task_id,
            "success": success,
            "total_steps": total_steps,
            "eval_score": eval_score,
            "sites": sites_raw if isinstance(sites_raw, list)
                     else [sites_raw] if sites_raw else [],
            "stop_reason": stop_reason,
            "model": model,
            "intent": intent,
            "agent_variant": agent_variant,
            "condition": condition,
            "experiment_id": experiment_id,
            "steps": data.get("steps", []),  # needed for token aggregation
        })

        if stop_reason in PIPELINE_ANOMALY_REASONS:
            skipped_rows.append({
                "task_id": task_id,
                "trajectory_path": traj_path,
                "stop_reason": stop_reason,
                "reason": "pipeline_anomaly",
            })

    return manifest_rows, result_dicts, skipped_rows


def write_manifest_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_skipped_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SKIPPED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_results_json(
    results: list[dict],
    path: Path,
    *,
    condition: str,
    experiment_id: str,
    data_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition": condition,
        "experiment_id": experiment_id,
        "source_dir": str(data_dir),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tasks": len(results),
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def print_summary(
    manifest_rows: list[dict],
    skipped_rows: list[dict],
    out_dir: Path,
) -> None:
    n = len(manifest_rows)
    if n == 0:
        print("No trajectories scanned.")
        return

    n_success = sum(1 for r in manifest_rows if r["success"] is True)
    n_failure = sum(1 for r in manifest_rows if r["success"] is False)
    n_unscored = n - n_success - n_failure

    print(f"Trajectories scanned: {n}")
    print(f"  success:   {n_success}")
    print(f"  failure:   {n_failure}")
    print(f"  unscored:  {n_unscored}  (success is null -- crash / evaluator error)")
    print(f"Pipeline anomalies (skipped.csv): {len(skipped_rows)}")

    bin_counts: dict[str, int] = {}
    for r in manifest_rows:
        bin_counts[r["length_bin"]] = bin_counts.get(r["length_bin"], 0) + 1
    print("Bin distribution:")
    for b in ("<=10", "11-20", "21-40", "41-80", "out_of_range"):
        if b in bin_counts:
            pct = 100 * bin_counts[b] / n
            print(f"  {b:<14} {bin_counts[b]:>4}  ({pct:.1f}%)")

    stop_counts: dict[str, int] = {}
    for r in manifest_rows:
        stop_counts[r["stop_reason"]] = stop_counts.get(r["stop_reason"], 0) + 1
    print("Stop-reason distribution:")
    for reason in sorted(stop_counts):
        print(f"  {reason:<18} {stop_counts[reason]:>4}")

    print()
    print(f"Wrote manifest to {out_dir / 'manifest.csv'}")
    print(f"Wrote results to  {out_dir / 'results.json'}")
    print(f"Wrote skipped to  {out_dir / 'skipped.csv'}")


def run_build(
    *,
    data_dir: Path,
    out_dir: Path,
    condition: str,
    experiment_id: str,
) -> int:
    """End-to-end manifest build. Returns process exit code."""
    if not data_dir.is_dir():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        return 1

    manifest_rows, result_dicts, skipped_rows = build_manifest_rows(
        data_dir, condition, experiment_id
    )

    write_manifest_csv(manifest_rows, out_dir / "manifest.csv")
    write_results_json(
        result_dicts,
        out_dir / "results.json",
        condition=condition,
        experiment_id=experiment_id,
        data_dir=data_dir,
    )
    write_skipped_csv(skipped_rows, out_dir / "skipped.csv")

    print_summary(manifest_rows, skipped_rows, out_dir)
    return 0
