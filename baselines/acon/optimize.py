"""CLI: optimize compression guidelines from offline JSONL pair records."""

from __future__ import annotations

import argparse
import os

from .guideline_update import optimize_guidelines_round
from .io import (
    load_guideline_state,
    load_pair_records_jsonl,
    resolve_full_trajectory_text,
    save_guideline_state,
)
from .templating import build_guideline_update_user_message
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ACON-style guideline optimization from JSONL pair records."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to JSONL file; one PairRecord object per line (see baseline_notes.txt).",
    )
    parser.add_argument(
        "--guideline-in",
        required=True,
        help="Input guideline JSON (text, version, meta).",
    )
    parser.add_argument(
        "--guideline-out",
        required=True,
        help="Output path for updated guideline JSON.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ACON_MODEL", ""),
        help="OpenRouter model id (default: env ACON_MODEL or llm.py default).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts for qualifying pairs; no LLM calls; no output file.",
    )
    args = parser.parse_args(argv)

    state = load_guideline_state(args.guideline_in)
    pairs = load_pair_records_jsonl(args.pairs)

    if args.dry_run:
        for i, pair in enumerate(pairs):
            if not pair.should_update_guideline():
                print(
                    f"[skip] pair {i} task_id={pair.task_id!r} "
                    f"(need full_success=true and compressed_success=false)"
                )
                continue
            full_text = resolve_full_trajectory_text(pair)
            msg = build_guideline_update_user_message(
                guideline=state.text,
                intent=pair.intent,
                full_trajectory_or_summary=full_text,
                compressed_context_used=pair.compressed_context_used or "",
                failure_signal=pair.failure_signal
                or f"full_success={pair.full_success}, "
                f"compressed_success={pair.compressed_success}",
            )
            print(f"--- pair {i} task_id={pair.task_id!r} ---")
            print(msg)
            print()
        return 0

    new_state = optimize_guidelines_round(
        pairs, state, model=args.model or os.environ.get("ACON_MODEL", "")
    )
    save_guideline_state(args.guideline_out, new_state)
    print(f"Wrote guideline version {new_state.version} to {args.guideline_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
