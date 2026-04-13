"""Parse LLM outputs with explicit section markers."""

from __future__ import annotations

COMPRESSED_BEGIN = "### COMPRESSED_CONTEXT"
GUIDELINE_BEGIN = "### REVISED_GUIDELINE"


def _after_marker(text: str, marker: str) -> str:
    if marker not in text:
        return text.strip()
    _, _, rest = text.partition(marker)
    return rest.strip()


def extract_compressed_context(raw_response: str) -> str:
    return _after_marker(raw_response, COMPRESSED_BEGIN)


def extract_revised_guideline(raw_response: str) -> str:
    return _after_marker(raw_response, GUIDELINE_BEGIN)
