"""
Perfect-retrieval baseline (oracle): embed per-step trajectory chunks from the
past only, query with the logged ground-truth action at the current step, and
return the top-k chunks by cosine similarity.

This is an upper bound: a real system cannot use the future logged action as the
query.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from baselines.trajectory import DEFAULT_ENCODING_NAME, count_tokens, format_step_line, get_encoding

EmbedBatchFn = Callable[[list[str]], np.ndarray]


def chunk_text_for_step(step: dict[str, Any], *, include_url: bool = True) -> str:
    """
    One embeddable chunk per SPEC step: same line as format_step_line, plus
    optional url on its own line for navigation grounding.
    """
    line = format_step_line(step)
    if include_url:
        url = step.get("url") or ""
        if str(url).strip():
            return f"{line}\nurl={url}"
    return line


@dataclass(frozen=True)
class ChunkRecord:
    """Single trajectory step indexed by SPEC step number t (1-based)."""

    t: int
    text: str


@dataclass(frozen=True)
class PerfectRetrievalResult:
    """Outcome of one oracle retrieval call at decision time for SPEC step t."""

    t_query: int
    query_text: str
    retrieved_ts: tuple[int, ...]
    similarity_by_t: dict[int, float]
    assembled_context: str
    context_token_count: int


def steps_to_chunks(
    steps: Sequence[dict[str, Any]], *, include_url: bool = True
) -> list[ChunkRecord]:
    return [
        ChunkRecord(t=int(s["t"]), text=chunk_text_for_step(s, include_url=include_url))
        for s in steps
    ]


def _ground_truth_query_text(steps: list[dict[str, Any]], t: int) -> str:
    """Text used as retrieval query: logged action at step t (oracle)."""
    if t < 1 or t > len(steps):
        raise ValueError(f"t must be in [1, len(steps)]; got t={t}, len={len(steps)}")
    row = steps[t - 1]
    action = (row.get("action") or "").strip()
    if action:
        return action
    thought = (row.get("thought") or "").strip()
    if thought:
        return thought
    rp = (row.get("raw_prediction") or "").strip()
    if rp:
        return rp[:2000]
    return ""


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return mat / norms


def perfect_retrieve_context(
    steps: list[dict[str, Any]],
    t: int,
    k: int,
    embed_fn: EmbedBatchFn,
    *,
    include_url: bool = True,
    encoding_name: str = DEFAULT_ENCODING_NAME,
) -> PerfectRetrievalResult:
    """
    At simulated decision time for SPEC step ``t`` (1-based):

    - **Corpus**: one chunk per prior step with ``step['t'] < t`` (strict past).
    - **Query**: embedding of the logged ground-truth for this step (``action``,
      else ``thought``, else truncated ``raw_prediction``) — oracle.

    Returns the top-``k`` past chunks by cosine similarity, assembled in
    chronological order (increasing ``t``).
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    query_text = _ground_truth_query_text(steps, t)

    past = [c for c in steps_to_chunks(steps, include_url=include_url) if c.t < t]
    if not past or k == 0:
        enc = get_encoding(encoding_name)
        assembled = ""
        return PerfectRetrievalResult(
            t_query=t,
            query_text=query_text,
            retrieved_ts=(),
            similarity_by_t={},
            assembled_context=assembled,
            context_token_count=count_tokens(assembled, enc),
        )

    corpus_texts = [c.text for c in past]

    q_vec = _normalize_rows(embed_fn([query_text]))[0]
    c_mat = _normalize_rows(embed_fn(corpus_texts))
    sims = c_mat @ q_vec

    order = np.argsort(-sims)
    top_idx = order[: min(k, len(order))]

    selected: list[tuple[int, float, str]] = []
    for i in top_idx:
        idx = int(i)
        ti = past[idx].t
        selected.append((ti, float(sims[idx]), past[idx].text))

    selected.sort(key=lambda x: x[0])
    retrieved_ts = tuple(x[0] for x in selected)
    similarity_by_t = {x[0]: x[1] for x in selected}
    assembled = "\n\n".join(x[2] for x in selected)

    enc = get_encoding(encoding_name)
    return PerfectRetrievalResult(
        t_query=t,
        query_text=query_text,
        retrieved_ts=retrieved_ts,
        similarity_by_t=similarity_by_t,
        assembled_context=assembled,
        context_token_count=count_tokens(assembled, enc),
    )


def deterministic_embed_fn(dim: int = 128) -> EmbedBatchFn:
    """
    Deterministic pseudo-embeddings for tests and dry runs (no API keys).

    Each string is mapped to a unit-norm vector in R^dim via SHA256-seeded RNG.
    """

    def _vec_for_text(text: str) -> np.ndarray:
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()[:8]
        seed = struct.unpack("<Q", seed_bytes)[0] % (2**32 - 1)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim, dtype=np.float64)
        n = np.linalg.norm(v)
        if n == 0.0:
            return np.zeros(dim, dtype=np.float64)
        return v / n

    def _embed(texts: list[str]) -> np.ndarray:
        return np.stack([_vec_for_text(t) for t in texts], axis=0)

    return _embed
