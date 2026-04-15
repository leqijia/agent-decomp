"""Automated first-pass review of oracle output files.

Loads all files in oracle/outputs/ (excluding 999_* and dummy_*),
inspects each parsed oracle state, flags suspicious outputs, and
prints a summary table.

Flags:
  P_EMPTY   — P_t is empty at step > 5 (agent should have done something)
  F_NONEMPTY — F_t is non-empty (failure mode present — worth inspecting)
  R_EMPTY   — R_t is empty (no remaining steps means task done or oracle confused)
  E_SHORT   — e_t is under 50 chars (probably truncated or malformed)
  K_EMPTY   — K_t is empty (no grounded facts extracted)
  NO_PARSE  — parsed field is None (raw_response was not valid JSON)

Usage:
    python oracle/review_oracle_outputs.py
    python oracle/review_oracle_outputs.py --update-notes   # also rewrites quality_notes.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

OUTPUTS_DIR = Path(__file__).parent / "outputs"
QUALITY_NOTES_PATH = Path(__file__).parent / "quality_notes.md"

E_T_SHORT_THRESHOLD = 50
P_T_EMPTY_AFTER_STEP = 5


def _load_outputs(outputs_dir: Path) -> list[dict]:
    """Load all oracle output files, skipping test fixtures and ablations."""
    records = []
    for fpath in sorted(outputs_dir.glob("*.json")):
        name = fpath.name
        if name.startswith("999_") or name.startswith("dummy_"):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: could not read {fpath.name}: {e}", file=sys.stderr)
            continue
        records.append({"file": name, "data": data})
    return records


def _flags(record: dict) -> list[str]:
    data = record["data"]
    parsed = data.get("parsed")
    step = data.get("step", 0)

    if parsed is None:
        return ["NO_PARSE"]

    issues = []

    p_t = parsed.get("P_t", [])
    r_t = parsed.get("R_t", [])
    e_t = parsed.get("e_t", "")
    f_t = parsed.get("F_t", [])
    k_t = parsed.get("K_t", [])

    if len(p_t) == 0 and step > P_T_EMPTY_AFTER_STEP:
        issues.append("P_EMPTY")
    if len(f_t) > 0:
        issues.append("F_NONEMPTY")
    if len(r_t) == 0:
        issues.append("R_EMPTY")
    if len(e_t) < E_T_SHORT_THRESHOLD:
        issues.append("E_SHORT")
    if len(k_t) == 0:
        issues.append("K_EMPTY")

    return issues


def _fmt_count(items) -> str:
    if items is None:
        return "?"
    if isinstance(items, int):
        return str(items)
    return str(len(items))


def review(outputs_dir: Path) -> list[dict]:
    records = _load_outputs(outputs_dir)
    results = []
    for rec in records:
        data = rec["data"]
        parsed = data.get("parsed") or {}
        step = data.get("step", "?")
        task_id = data.get("trajectory_id", "?")

        p_t = parsed.get("P_t")
        r_t = parsed.get("R_t")
        e_t = parsed.get("e_t", "")
        f_t = parsed.get("F_t")
        k_t = parsed.get("K_t")

        issues = _flags(rec)

        results.append({
            "file": rec["file"],
            "task_id": task_id,
            "step": step,
            "p_t_count": len(p_t) if p_t is not None else None,
            "r_t_count": len(r_t) if r_t is not None else None,
            "e_t_len": len(e_t),
            "f_t_count": len(f_t) if f_t is not None else None,
            "k_t_count": len(k_t) if k_t is not None else None,
            "flags": issues,
        })
    return results


def print_table(results: list[dict]) -> None:
    header = f"{'Task':<8} {'Step':<6} {'P_t':>4} {'R_t':>4} {'e_t_len':>8} {'F_t':>4} {'K_t':>4}  Flags"
    print(header)
    print("-" * len(header))
    for r in results:
        flags_str = ", ".join(r["flags"]) if r["flags"] else "ok"
        print(
            f"{str(r['task_id']):<8} "
            f"{str(r['step']):<6} "
            f"{_fmt_count(r['p_t_count']):>4} "
            f"{_fmt_count(r['r_t_count']):>4} "
            f"{r['e_t_len']:>8} "
            f"{_fmt_count(r['f_t_count']):>4} "
            f"{_fmt_count(r['k_t_count']):>4}  "
            f"{flags_str}"
        )


def print_summary(results: list[dict]) -> None:
    total = len(results)
    with_flags = sum(1 for r in results if r["flags"])
    clean = total - with_flags
    f_t_nonempty = sum(1 for r in results if "F_NONEMPTY" in r["flags"])
    p_suspicious = sum(1 for r in results if "P_EMPTY" in r["flags"])
    no_parse = sum(1 for r in results if "NO_PARSE" in r["flags"])

    print()
    print(f"Files reviewed:       {total}")
    print(f"Clean (no flags):     {clean}")
    print(f"Suspicious P_t:       {p_suspicious}  (empty after step {P_T_EMPTY_AFTER_STEP})")
    print(f"Non-empty F_t:        {f_t_nonempty}")
    print(f"Parse failures:       {no_parse}")
    print(f"Total with any flag:  {with_flags}")


def _build_quality_notes_section(results: list[dict]) -> str:
    flagged = [r for r in results if r["flags"]]
    clean = [r for r in results if not r["flags"]]

    lines = ["## Auto-review results\n"]
    lines.append(f"Generated by `oracle/review_oracle_outputs.py` — {len(results)} files reviewed.\n")

    lines.append("\n### Flagged outputs\n")
    if flagged:
        lines.append("| Task | Step | Flags |")
        lines.append("|------|------|-------|")
        for r in flagged:
            lines.append(f"| {r['task_id']} | {r['step']} | {', '.join(r['flags'])} |")
    else:
        lines.append("None — all outputs passed automated checks.")

    lines.append("\n### Clean outputs\n")
    if clean:
        ids = sorted({str(r["task_id"]) for r in clean})
        lines.append(f"Tasks with all steps clean: {', '.join(ids)}")
    else:
        lines.append("None passed without flags.")

    lines.append("\n### Flag legend\n")
    lines.append("- **P_EMPTY** — P_t empty at step > 5 (agent should have made progress)")
    lines.append("- **F_NONEMPTY** — F_t contains failure modes (inspect manually)")
    lines.append("- **R_EMPTY** — R_t empty (task done, or oracle confused about remaining steps)")
    lines.append("- **E_SHORT** — e_t under 50 chars (likely truncated)")
    lines.append("- **K_EMPTY** — K_t empty (no grounded facts extracted)")
    lines.append("- **NO_PARSE** — raw_response is not valid JSON\n")

    return "\n".join(lines)


def update_quality_notes(results: list[dict], notes_path: Path) -> None:
    with open(notes_path, encoding="utf-8") as f:
        existing = f.read()

    section = _build_quality_notes_section(results)
    marker = "## Auto-review results"

    if marker in existing:
        # Replace from the marker to the end of the file
        idx = existing.index(marker)
        updated = existing[:idx] + section
    else:
        updated = existing.rstrip("\n") + "\n\n" + section

    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(updated)

    print(f"\nUpdated {notes_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review oracle output files for quality issues.")
    parser.add_argument(
        "--update-notes", action="store_true",
        help="Rewrite the ## Auto-review results section in oracle/quality_notes.md",
    )
    parser.add_argument(
        "--outputs-dir", default=str(OUTPUTS_DIR),
        help="Path to oracle outputs directory",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"ERROR: outputs dir not found: {outputs_dir}", file=sys.stderr)
        sys.exit(1)

    results = review(outputs_dir)

    if not results:
        print("No oracle output files found (excluding 999_* and dummy_*).")
        return

    print_table(results)
    print_summary(results)

    if args.update_notes:
        update_quality_notes(results, QUALITY_NOTES_PATH)


if __name__ == "__main__":
    main()
