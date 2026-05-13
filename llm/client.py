"""Unified LLM dispatch.

Routes a chat_completion call to the right provider based on the model
string. Gemini models go through the Google API (GEMINI_API_KEY);
everything else goes through OpenRouter (OPENROUTER_API_KEY).

Callers import this instead of choosing a provider directly:

    from llm.client import chat_completion
    result = chat_completion(messages, model=os.environ["AGENT_MODEL"], ...)
"""
from __future__ import annotations

from typing import Any

from llm.openrouter import ChatResult


def _is_gemini(model: str) -> bool:
    return model.startswith("gemini-") or model.startswith("models/gemini-")


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = 2048,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    timeout_s: int = 60,
    **kwargs: Any,
) -> ChatResult:
    if _is_gemini(model):
        from llm.gemini import chat_completion as _gemini_chat
        return _gemini_chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p, stop=stop,
            timeout_s=timeout_s, **kwargs,
        )
    else:
        from llm.openrouter import chat_completion as _openrouter_chat
        return _openrouter_chat(
            messages, model=model, temperature=temperature,
            max_tokens=max_tokens, top_p=top_p, stop=stop,
            timeout_s=timeout_s, **kwargs,
        )
