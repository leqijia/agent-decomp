"""Propose guideline updates from paired success/failure records (ACON-style)."""

from __future__ import annotations

from baselines.trajectory import DEFAULT_ENCODING_NAME, count_tokens, get_encoding

from .io import default_failure_signal, resolve_full_trajectory_text
from .llm import LLMGenerate, openrouter_generate
from .parsing import extract_revised_guideline
from .templating import build_guideline_update_user_message
from .types import GuidelineState, PairRecord


def propose_guideline_update(
    pair: PairRecord,
    state: GuidelineState,
    *,
    llm: LLMGenerate | None = None,
    model: str = "",
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> tuple[str, str]:
    """
    Returns (revised_guideline_text, raw_llm_response).

    Skips the LLM call when ``pair.should_update_guideline()`` is False;
    in that case returns ``(state.text, "")``.
    """
    if not pair.should_update_guideline():
        return state.text, ""

    full_text = resolve_full_trajectory_text(pair)
    compressed = pair.compressed_context_used or ""
    failure = default_failure_signal(pair)

    user_message = build_guideline_update_user_message(
        guideline=state.text,
        intent=pair.intent,
        full_trajectory_or_summary=full_text,
        compressed_context_used=compressed,
        failure_signal=failure,
    )

    enc = get_encoding(encoding_name)
    _ = count_tokens(user_message, enc)

    generate = llm if llm is not None else openrouter_generate
    if generate is openrouter_generate:
        raw = openrouter_generate(user_message, model=model or None)
    else:
        raw = generate(user_message)

    revised = extract_revised_guideline(raw).strip()
    if not revised:
        revised = state.text
    return revised, raw


def optimize_guidelines_round(
    pairs: list[PairRecord],
    state: GuidelineState,
    *,
    llm: LLMGenerate | None = None,
    model: str = "",
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> GuidelineState:
    """
    Sequentially apply guideline updates for each pair that qualifies.

    Version increments once per pair processed (including skipped pairs if you
    want strict counting — here we increment only when the guideline text
    changes or an LLM call was made).
    """
    text = state.text
    version = state.version
    meta = dict(state.meta)
    raw_responses: list[str] = []

    for pair in pairs:
        new_text, raw = propose_guideline_update(
            pair,
            GuidelineState(text=text, version=version, meta=meta),
            llm=llm,
            model=model,
            encoding_name=encoding_name,
        )
        if raw:
            raw_responses.append(raw)
        if new_text != text:
            version += 1
            text = new_text

    meta["last_round_raw_responses"] = raw_responses
    return GuidelineState(text=text, version=version, meta=meta)
