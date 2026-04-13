"""Load/save guideline state, pair records (JSONL), and trajectory JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from baselines.trajectory import serialize_trajectory

from .types import GuidelineState, PairRecord


def load_trajectory_steps(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"{path}: expected top-level 'steps' list")
    return steps


def trajectory_to_text(path: str | Path) -> str:
    return serialize_trajectory(load_trajectory_steps(path))


def load_guideline_state(path: str | Path) -> GuidelineState:
    with open(path, encoding="utf-8") as f:
        return GuidelineState.from_json(json.load(f))


def save_guideline_state(path: str | Path, state: GuidelineState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state.to_json(), f, indent=2)


def pair_record_from_dict(d: dict[str, Any]) -> PairRecord:
    return PairRecord(
        task_id=d["task_id"],
        intent=d["intent"],
        full_success=bool(d["full_success"]),
        compressed_success=bool(d["compressed_success"]),
        path_full_traj=d.get("path_full_traj"),
        path_compressed_traj=d.get("path_compressed_traj"),
        full_trajectory_text=d.get("full_trajectory_text"),
        compressed_context_used=d.get("compressed_context_used"),
        failure_signal=d.get("failure_signal"),
    )


def load_pair_records_jsonl(path: str | Path) -> list[PairRecord]:
    records: list[PairRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(pair_record_from_dict(json.loads(line)))
    return records


def iter_pair_records_jsonl(path: str | Path) -> Iterator[PairRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield pair_record_from_dict(json.loads(line))


def resolve_full_trajectory_text(pair: PairRecord) -> str:
    if pair.full_trajectory_text:
        return pair.full_trajectory_text
    if pair.path_full_traj:
        return trajectory_to_text(pair.path_full_traj)
    raise ValueError(
        f"Pair task_id={pair.task_id!r}: need full_trajectory_text or path_full_traj"
    )


def default_failure_signal(pair: PairRecord) -> str:
    if pair.failure_signal:
        return pair.failure_signal
    return (
        f"full_success={pair.full_success}, compressed_success={pair.compressed_success}"
    )
