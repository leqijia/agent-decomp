"""Google Gemini API client.

Same interface as openrouter.py (returns ChatResult) so callers don't need
to know which provider is backing the model. Uses the GEMINI_API_KEY from
.env and the google-generativeai SDK.
"""
from __future__ import annotations

import os
import time
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv

from llm.openrouter import ChatResult, OpenRouterError

load_dotenv()


def _configure() -> None:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise OpenRouterError(
            "GEMINI_API_KEY is not set. Put it in .env or export it."
        )
    genai.configure(api_key=key)


_configured = False


def _ensure_configured() -> None:
    global _configured
    if not _configured:
        _configure()
        _configured = True


def _messages_to_gemini(messages: list[dict[str, Any]]) -> tuple[str | None, list[dict]]:
    """Convert OpenAI chat-messages format to Gemini content format.

    Returns (system_instruction, contents) where contents is a list of
    {"role": "user"|"model", "parts": [text]} dicts.
    """
    system = None
    contents = []
    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            # Gemini supports system instructions as a separate parameter.
            # If there are multiple system messages (e.g. intro + examples),
            # concatenate them.
            system = f"{system}\n\n{text}" if system else text
        elif role == "user":
            contents.append({"role": "user", "parts": [text]})
        elif role == "assistant":
            contents.append({"role": "model", "parts": [text]})
    return system, contents


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    timeout_s: int = 60,
    **kwargs: Any,
) -> ChatResult:
    """Make one chat call to the Gemini API.

    Accepts the same OpenAI chat-messages format as openrouter.chat_completion
    and returns the same ChatResult, so the agent loop doesn't need to know
    which provider is in use.
    """
    _ensure_configured()

    system_instruction, contents = _messages_to_gemini(messages)

    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        top_p=top_p,
    )
    if max_tokens is not None:
        generation_config.max_output_tokens = max_tokens
    if stop:
        generation_config.stop_sequences = stop

    gen_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_instruction,
        generation_config=generation_config,
    )

    t0 = time.monotonic()
    try:
        response = gen_model.generate_content(
            contents,
            request_options={"timeout": timeout_s},
        )
    except Exception as e:
        raise OpenRouterError(f"Gemini API request failed: {e}") from e
    latency_ms = int((time.monotonic() - t0) * 1000)

    try:
        content = response.text
    except ValueError as e:
        # Gemini may block content or return empty; surface it clearly
        raise OpenRouterError(
            f"Gemini returned no text. "
            f"Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}. "
            f"Safety: {response.prompt_feedback}"
        ) from e

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    return ChatResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        raw={"model": model, "usage": str(usage)},
    )
