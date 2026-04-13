"""Load and fill ACON-style prompt templates."""

from __future__ import annotations

import os
from typing import Any


def _prompts_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "prompts")


def load_template(name: str) -> str:
    path = os.path.join(_prompts_dir(), f"{name}.txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


def fill_template(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)


def build_compress_user_message(
    *,
    intent: str,
    guideline: str,
    trajectory_text: str,
    template: str | None = None,
) -> str:
    t = template if template is not None else load_template("compress_user")
    return fill_template(
        t, intent=intent, guideline=guideline, trajectory_text=trajectory_text
    )


def build_guideline_update_user_message(
    *,
    guideline: str,
    intent: str,
    full_trajectory_or_summary: str,
    compressed_context_used: str,
    failure_signal: str,
    template: str | None = None,
) -> str:
    t = template if template is not None else load_template("guideline_update_user")
    return fill_template(
        t,
        guideline=guideline,
        intent=intent,
        full_trajectory_or_summary=full_trajectory_or_summary,
        compressed_context_used=compressed_context_used,
        failure_signal=failure_signal,
    )
