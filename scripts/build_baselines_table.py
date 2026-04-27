"""Aggregate per-condition metrics.json into the deliverable baselines table.

Reads metrics.json for the raw + self_generated Exp3 conditions and the five
baselines (sliding_window, observation_masking, acon, agentdiet,
perfect_retrieval). Emits two artifacts under `experiments/baselines/`:

  baselines_table.md   -- markdown table for the writeup
  baselines_table.json -- structured form for downstream plotting / regen

Missing inputs are skipped with a warning (so the table can be regenerated
mid-experiment as conditions land). Token figures come from
metrics["tokens"]["avg_tokens"] (added by
`scripts/compute_condition_metrics.py`); rows that lack it (e.g. an older
metrics.json from before the avg_tokens upgrade) report `n/a`.

Usage:
    python scripts/build_baselines_table.py
    python scripts/build_baselines_table.py \\
        --out-md   experiments/baselines/baselines_table.md \\
        --out-json experiments/baselines/baselines_table.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.metrics import LENGTH_BINS


METHOD_ORDER: list[tuple[str, str]] = [
    ("raw",                 "experiments/exp3/raw/metrics.json"),
    ("self_generated",      "experiments/exp3/self_generated/metrics.json"),
    ("sliding_window",      "experiments/baselines/sliding_window/metrics.json"),
    ("observation_masking", "experiments/baselines/observation_masking/metrics.json"),
    ("acon",                "experiments/baselines/acon/metrics.json"),
    ("agentdiet",           "experiments/baselines/agentdiet/metrics.json"),
    ("perfect_retrieval",   "experiments/baselines/perfect_retrieval/metrics.json"),
]


def _fmt_pct(x: float | None) -> str:
    return f"{100 * x:.1f}%" if x is not None else "n/a"


def _fmt_ci(ci) -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    return f"[{100 * ci[0]:.1f}%, {100 * ci[1]:.1f}%]"


def _fmt_int(x) -> str:
    return f"{int(x):,}" if x is not None else "n/a"


def _fmt_float(x, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if x is not None else "n/a"


def _load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        print(f"  WARNING: metrics file missing, skipping: {path}", file=sys.stderr)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: could not load {path}: {e}", file=sys.stderr)
        return None


def _row_for_method(label: str, metrics: dict) -> dict:
    overall = metrics.get("overall", {}) or {}
    tokens = metrics.get("tokens", {}) or {}
    es = metrics.get("eval_score", {}) or {}
    by_bin = metrics.get("by_length_bin", {}) or {}
    return {
        "method":         label,
        "n_tasks":        metrics.get("n_tasks"),
        "n_scored":       overall.get("n"),
        "n_unscored":     overall.get("n_unscored"),
        "success_rate":   overall.get("success_rate"),
        "ci_95":          overall.get("ci_95"),
        "avg_tokens":     tokens.get("avg_tokens"),
        "avg_eval_score": es.get("avg_eval_score"),
        "by_length_bin":  {
            b: by_bin.get(b, {}).get("success_rate") for b in LENGTH_BINS
        },
        "model": metrics.get("model"),
        "source_metrics": metrics.get("source_results", ""),
    }


def build_table_payload(layout: list[tuple[str, str]]) -> dict:
    rows: list[dict] = []
    missing: list[str] = []
    for label, rel in layout:
        m = _load_metrics(Path(rel))
        if m is None:
            missing.append(label)
            continue
        rows.append(_row_for_method(label, m))
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "length_bins": LENGTH_BINS,
        "rows":        rows,
        "missing":     missing,
    }


def render_markdown(payload: dict) -> str:
    bins = payload["length_bins"]
    rows = payload["rows"]

    out: list[str] = []
    out.append("# Baselines table")
    out.append("")
    out.append(f"_Generated: {payload['generated_at']}_")
    if payload["missing"]:
        out.append("")
        out.append(f"**Missing methods (no metrics.json found):** "
                   f"{', '.join(payload['missing'])}")
    out.append("")

    out.append("## Headline (overall)")
    out.append("")
    out.append("| Method | n | Success rate | 95% CI | Avg tokens / task | Avg eval_score |")
    out.append("|---|---:|---:|---|---:|---:|")
    for r in rows:
        out.append(
            f"| {r['method']} "
            f"| {_fmt_int(r['n_scored'])} "
            f"| {_fmt_pct(r['success_rate'])} "
            f"| {_fmt_ci(r['ci_95'])} "
            f"| {_fmt_int(r['avg_tokens'])} "
            f"| {_fmt_float(r['avg_eval_score'])} |"
        )
    out.append("")

    out.append("## By length bin (success rate)")
    out.append("")
    header = "| Method | " + " | ".join(bins) + " |"
    sep = "|---|" + "|".join(["---:"] * len(bins)) + "|"
    out.append(header)
    out.append(sep)
    for r in rows:
        cells = [_fmt_pct(r["by_length_bin"].get(b)) for b in bins]
        out.append(f"| {r['method']} | " + " | ".join(cells) + " |")
    out.append("")

    out.append("## Notes")
    out.append("")
    out.append("- `n` is the count of *scored* tasks "
               "(tasks where evaluation succeeded; crashes are reported "
               "separately as unscored in the per-condition metrics.json).")
    out.append("- 95% CIs are Wilson-score intervals on the per-method success "
               "rate (`harness.metrics._wilson_ci`).")
    out.append("- `Avg tokens / task` sums prompt + completion tokens across "
               "every step in every trajectory and divides by tasks.")
    out.append("- `perfect_retrieval` uses live-time query semantics "
               "(`intent + current_observation`), not the strict-oracle query "
               "from `baselines/perfect_retrieval.py`. See "
               "`experiments/baselines/README.md`.")

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate per-condition metrics.json files into the baselines deliverable."
    )
    parser.add_argument(
        "--out-md", default="experiments/baselines/baselines_table.md",
        help="Output markdown table path.",
    )
    parser.add_argument(
        "--out-json", default="experiments/baselines/baselines_table.json",
        help="Output structured JSON path.",
    )
    args = parser.parse_args()

    payload = build_table_payload(METHOD_ORDER)
    if not payload["rows"]:
        print("ERROR: no metrics.json files found for any method. "
              "Run scripts/compute_condition_metrics.py first.", file=sys.stderr)
        return 1

    md = render_markdown(payload)

    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")
    print(f"Methods included: {[r['method'] for r in payload['rows']]}")
    if payload["missing"]:
        print(f"Methods skipped (no metrics.json): {payload['missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
