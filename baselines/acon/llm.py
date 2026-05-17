"""Optional OpenRouter chat helper; tests should inject a stub instead."""

from __future__ import annotations

import os
import sys
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from llm.config import ACON_MODEL
from llm.client import chat_completion
from llm.openrouter import ChatResult

LLMGenerate = Callable[[str], str]


def openrouter_generate(
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    timeout_s: int = 120,
) -> str:
    return openrouter_completion(
        user_message,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
    ).content.strip()


def openrouter_completion(
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    timeout_s: int = 120,
) -> ChatResult:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; pass llm= to compress_trajectory / "
            "propose_guideline_update or set the environment variable."
        )
    model = model or ACON_MODEL
    return chat_completion(
        [{"role": "user", "content": user_message}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )
