"""LLM trajectory compression conditioned on a natural-language guideline."""

from __future__ import annotations

import os
from typing import Any, Callable

from baselines.trajectory import (
    DEFAULT_ENCODING_NAME,
    count_tokens,
    get_encoding,
    serialize_trajectory,
)

from llm.config import ACON_MODEL
from .llm import LLMGenerate, openrouter_completion, openrouter_generate
from .parsing import extract_compressed_context
from .templating import build_compress_user_message
from .types import CompressedContext


def compress_trajectory(
    intent: str,
    steps: list[dict[str, Any]],
    guideline: str,
    *,
    llm: LLMGenerate | None = None,
    model: str = "",
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> CompressedContext:
    """
    Compress serialized trajectory text using the given guideline.

    If ``llm`` is None, calls OpenRouter via ``openrouter_generate`` (requires
    ``OPENROUTER_API_KEY`` and optional ``ACON_MODEL``).
    """
    trajectory_text = serialize_trajectory(steps)
    user_message = build_compress_user_message(
        intent=intent, guideline=guideline, trajectory_text=trajectory_text
    )
    enc = get_encoding(encoding_name)
    input_tokens = count_tokens(user_message, enc)
    output_tokens = 0

    generate = llm if llm is not None else openrouter_generate
    if generate is openrouter_generate:
        chat = openrouter_completion(user_message, model=model or None)
        raw = chat.content.strip()
        used_model = model or ACON_MODEL
        input_tokens = chat.prompt_tokens or input_tokens
        output_tokens = chat.completion_tokens or 0
        cost_usd = chat.cost_usd
    else:
        raw = generate(user_message)
        used_model = model or "stub"
        cost_usd = None

    compressed = extract_compressed_context(raw)
    out_tokens = output_tokens or count_tokens(compressed, enc)
    return CompressedContext(
        compressed_text=compressed,
        raw_response=raw,
        input_tokens=input_tokens,
        output_tokens=out_tokens,
        model=used_model,
        cost_usd=cost_usd,
    )
