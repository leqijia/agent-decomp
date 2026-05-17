"""Datatypes for ACON-style compression and guideline optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompressedContext:
    compressed_text: str
    raw_response: str
    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float | None = None


@dataclass
class GuidelineState:
    text: str
    version: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"text": self.text, "version": self.version, "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> GuidelineState:
        return cls(
            text=d["text"],
            version=int(d.get("version", 0)),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class PairRecord:
    """One full-context vs compressed-context comparison for guideline updates."""

    task_id: str | int
    intent: str
    full_success: bool
    compressed_success: bool
    path_full_traj: str | None = None
    path_compressed_traj: str | None = None
    full_trajectory_text: str | None = None
    compressed_context_used: str | None = None
    failure_signal: str | None = None

    def should_update_guideline(self) -> bool:
        """ACON targets cases where full context works but compression hurts."""
        return bool(self.full_success) and not bool(self.compressed_success)
