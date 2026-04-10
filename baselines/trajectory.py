
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import tiktoken

WINDOW_4K = 4096
WINDOW_8K = 8192
WINDOW_16K = 16384

DEFAULT_ENCODING_NAME = "cl100k_base"
DEFAULT_OBSERVATION_PLACEHOLDER = "[OBSERVATION_REDACTED]"


def format_step_line(step: dict[str, Any]) -> str:
    return (
        f"Step {step['t']}: thought={step['thought']} | "
        f"action={step['action']} | observation={step['observation']}"
    )


def serialize_trajectory(steps: list[dict[str, Any]]) -> str:
    if not steps:
        return ""
    return "\n".join(format_step_line(s) for s in steps)


def get_encoding(encoding_name: str = DEFAULT_ENCODING_NAME) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: tiktoken.Encoding | None = None) -> int:
    enc = encoding or get_encoding()
    return len(enc.encode(text))


@dataclass(frozen=True)
class SlidingWindowResult:
    steps: list[dict[str, Any]]
    dropped_prefix_steps: int
    trajectory_token_count: int
    max_tokens: int
    encoding_name: str
    last_step_exceeds_budget: bool


def sliding_window_truncate(
    steps: list[dict[str, Any]],
    max_tokens: int,
    *,
    encoding_name: str = DEFAULT_ENCODING_NAME,
    encoding: tiktoken.Encoding | None = None,
) -> SlidingWindowResult:
    enc = encoding or tiktoken.get_encoding(encoding_name)
    lines = [format_step_line(s) for s in steps]
    if not lines:
        return SlidingWindowResult(
            steps=[],
            dropped_prefix_steps=0,
            trajectory_token_count=0,
            max_tokens=max_tokens,
            encoding_name=encoding_name,
            last_step_exceeds_budget=False,
        )

    n = len(lines)
    for k in range(n):
        suffix_text = "\n".join(lines[k:])
        tok_n = len(enc.encode(suffix_text))
        if tok_n <= max_tokens:
            return SlidingWindowResult(
                steps=steps[k:],
                dropped_prefix_steps=k,
                trajectory_token_count=tok_n,
                max_tokens=max_tokens,
                encoding_name=encoding_name,
                last_step_exceeds_budget=False,
            )

    last_only = "\n".join(lines[-1:])
    last_tok = len(enc.encode(last_only))
    return SlidingWindowResult(
        steps=steps[-1:],
        dropped_prefix_steps=n - 1,
        trajectory_token_count=last_tok,
        max_tokens=max_tokens,
        encoding_name=encoding_name,
        last_step_exceeds_budget=last_tok > max_tokens,
    )


MaskMode = Literal["all", "keep_last"]


def mask_observations(
    steps: list[dict[str, Any]],
    placeholder: str = DEFAULT_OBSERVATION_PLACEHOLDER,
    *,
    mask_mode: MaskMode = "all",
    step_index_in_placeholder: bool = False,
) -> list[dict[str, Any]]:
    """
    Replace step `observation` fields with a short placeholder; deep-copy steps.
    """
    out: list[dict[str, Any]] = []
    n = len(steps)
    for i, s in enumerate(steps):
        row = deepcopy(s)
        is_last = i == n - 1
        if mask_mode == "keep_last" and is_last:
            out.append(row)
            continue
        if step_index_in_placeholder:
            row["observation"] = f"{placeholder} step={row['t']}"
        else:
            row["observation"] = placeholder
        out.append(row)
    return out


def compose_mask_then_window(
    steps: list[dict[str, Any]],
    max_tokens: int,
    *,
    placeholder: str = DEFAULT_OBSERVATION_PLACEHOLDER,
    mask_mode: MaskMode = "all",
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> tuple[list[dict[str, Any]], SlidingWindowResult]:
    """Mask observations first, then sliding-window on the result."""
    masked = mask_observations(steps, placeholder=placeholder, mask_mode=mask_mode)
    result = sliding_window_truncate(masked, max_tokens, encoding_name=encoding_name)
    return result.steps, result


def compose_window_then_mask(
    steps: list[dict[str, Any]],
    max_tokens: int,
    *,
    placeholder: str = DEFAULT_OBSERVATION_PLACEHOLDER,
    mask_mode: MaskMode = "all",
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> tuple[list[dict[str, Any]], SlidingWindowResult]:
    """Sliding-window first, then mask observations on the retained suffix."""
    windowed = sliding_window_truncate(steps, max_tokens, encoding_name=encoding_name)
    masked = mask_observations(
        windowed.steps, placeholder=placeholder, mask_mode=mask_mode
    )
    return masked, SlidingWindowResult(
        steps=masked,
        dropped_prefix_steps=windowed.dropped_prefix_steps,
        trajectory_token_count=count_tokens(
            serialize_trajectory(masked),
            get_encoding(encoding_name),
        ),
        max_tokens=max_tokens,
        encoding_name=encoding_name,
        last_step_exceeds_budget=windowed.last_step_exceeds_budget,
    )
