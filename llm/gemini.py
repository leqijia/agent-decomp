"""Google Gemini API client via the google-genai SDK.

Same ChatResult interface as openrouter.py so callers don't need to know
which provider is backing the model. Uses GEMINI_API_KEY from .env.
"""
from __future__ import annotations

import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from llm.openrouter import ChatResult, OpenRouterError

load_dotenv()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise OpenRouterError(
                "GEMINI_API_KEY is not set. Put it in .env or export it."
            )
        _client = genai.Client(api_key=key)
    return _client


def _messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[types.Content]]:
    """Convert OpenAI chat-messages format to Gemini content format.

    Returns (system_instruction, contents). System messages become the
    system_instruction parameter; user/assistant messages become alternating
    Content objects with role "user" / "model".
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        elif role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part(text=text)]))

    system = "\n\n".join(system_parts) if system_parts else None
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
    and returns the same ChatResult so the agent loop is provider-agnostic.
    """
    client = _get_client()
    system_instruction, contents = _messages_to_gemini(messages)

    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens or 4096,
        system_instruction=system_instruction,
    )
    if stop:
        config.stop_sequences = stop

    t0 = time.monotonic()
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        raise OpenRouterError(f"Gemini API request failed: {e}") from e
    latency_ms = int((time.monotonic() - t0) * 1000)

    try:
        content = response.text
    except (ValueError, AttributeError):
        finish = (
            response.candidates[0].finish_reason
            if response.candidates
            else "unknown"
        )
        raise OpenRouterError(
            f"Gemini returned no text. Finish reason: {finish}. "
            f"Prompt feedback: {response.prompt_feedback}"
        )

    if content is None:
        raise OpenRouterError("Gemini returned None text content")

    usage = response.usage_metadata
    prompt_tokens = getattr(usage, "prompt_token_count", None) if usage else None
    completion_tokens = getattr(usage, "candidates_token_count", None) if usage else None

    return ChatResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        raw={"model": model, "usage": str(usage)},
    )
