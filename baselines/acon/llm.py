"""Optional OpenRouter chat helper; tests should inject a stub instead."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from llm.config import ACON_MODEL

LLMGenerate = Callable[[str], str]


def openrouter_generate(
    user_message: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    timeout_s: int = 120,
) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; pass llm= to compress_trajectory / "
            "propose_guideline_update or set the environment variable."
        )
    model = model or ACON_MODEL
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": user_message}],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e

    return body["choices"][0]["message"]["content"].strip()
