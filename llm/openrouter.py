"""OpenRouter chat-completions client.

Single entry point for every LLM call the project makes: the ReAct agent,
the fuzzy-match grader inside webarena/evaluation_harness, and eventually
the oracle generator. Routes through https://openrouter.ai/api/v1 and
returns usage metadata so callers can populate the per-step fields in
trajectories/SPEC.md (prompt_tokens, completion_tokens, latency_ms) and so
batch runs can track spend against the OPENROUTER budget.

Model slugs live in .env, not in code, so switching models is one edit.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_TIMEOUT_S = 60


@dataclass
class ChatResult:
    content: str                 # assistant message text
    prompt_tokens: int | None    # None iff provider didn't return usage
    completion_tokens: int | None
    latency_ms: int              # wall-clock of the HTTP call
    raw: dict[str, Any]          # full response body, for auditing
    cost_usd: float | None = None  # actual cost from OpenRouter usage.cost


class OpenRouterError(RuntimeError):
    """Raised on any non-2xx response or network failure. Do not swallow."""


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Put it in .env or export it."
        )
    return key


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    top_p: float = 1.0,
    stop: list[str] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> ChatResult:
    """Make one chat-completions call to OpenRouter.

    `messages` is the OpenAI chat format: list of {"role", "content"} dicts.
    `model` is an OpenRouter slug, e.g. "qwen/qwen3.5-27b". Caller is
    responsible for reading the slug from the environment if that is the
    desired source — this function does not fall back to a default.

    Raises OpenRouterError on HTTP errors, timeouts, or malformed responses.
    Does not retry; retry policy belongs to the caller (the ReAct loop
    retries parse failures at a higher layer).
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if stop is not None:
        payload["stop"] = stop
    if extra_body:
        payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }

    t0 = time.monotonic()
    try:
        resp = requests.post(_API_URL, headers=headers, json=payload, timeout=timeout_s)
    except requests.RequestException as e:
        raise OpenRouterError(f"OpenRouter request failed: {e}") from e
    latency_ms = int((time.monotonic() - t0) * 1000)

    if resp.status_code != 200:
        raise OpenRouterError(
            f"OpenRouter returned {resp.status_code}: {resp.text[:500]}"
        )

    try:
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as e:
        raise OpenRouterError(
            f"Malformed OpenRouter response: {resp.text[:500]}"
        ) from e

    usage = body.get("usage") or {}
    cost_raw = usage.get("cost")
    cost_usd = float(cost_raw) if cost_raw is not None else None
    return ChatResult(
        content=content,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_ms=latency_ms,
        raw=body,
        cost_usd=cost_usd,
    )


def simple_chat(
    prompt: str,
    *,
    model: str,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Convenience wrapper: one user prompt in, one string out.

    Used by the fuzzy-match grader in webarena/evaluation_harness, which does
    not care about usage metadata and only needs the assistant text.
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat_completion(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ).content
