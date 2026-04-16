"""Tests for perfect-retrieval oracle baseline."""

from __future__ import annotations

import unittest

from baselines.perfect_retrieval import (
    chunk_text_for_step,
    deterministic_embed_fn,
    perfect_retrieve_context,
    steps_to_chunks,
)


def _step(t: int, thought: str, action: str, observation: str, url: str = "") -> dict:
    d: dict = {
        "t": t,
        "thought": thought,
        "action": action,
        "observation": observation,
    }
    if url:
        d["url"] = url
    return d


class TestChunkText(unittest.TestCase):
    def test_matches_format_step_plus_url(self):
        s = _step(1, "a", "click [1]", "obs", url="http://x")
        txt = chunk_text_for_step(s)
        self.assertIn("Step 1:", txt)
        self.assertIn("click [1]", txt)
        self.assertIn("url=http://x", txt)

    def test_omit_url_when_disabled(self):
        s = _step(1, "a", "act", "o", url="http://x")
        txt = chunk_text_for_step(s, include_url=False)
        self.assertNotIn("url=", txt)


class TestNoFutureLeak(unittest.TestCase):
    def test_t1_empty_corpus(self):
        steps = [_step(1, "t1", "a1", "o1")]
        emb = deterministic_embed_fn(32)
        r = perfect_retrieve_context(steps, t=1, k=5, embed_fn=emb)
        self.assertEqual(r.assembled_context, "")
        self.assertEqual(r.retrieved_ts, ())
        self.assertEqual(r.query_text, "a1")

    def test_only_past_steps_in_topk(self):
        steps = [
            _step(1, "t1", "click [1]", "o1"),
            _step(2, "t2", "type [2] x", "o2"),
            _step(3, "t3", "stop [done]", "o3"),
        ]
        emb = deterministic_embed_fn(64)
        r = perfect_retrieve_context(steps, t=3, k=10, embed_fn=emb)
        for ti in r.retrieved_ts:
            self.assertLess(ti, 3)


class TestDeterministicRetrieval(unittest.TestCase):
    def test_k_limits_count(self):
        steps = [
            _step(1, "t1", "alpha", "o1"),
            _step(2, "t2", "beta", "o2"),
            _step(3, "t3", "gamma", "o3"),
        ]
        emb = deterministic_embed_fn(32)
        r = perfect_retrieve_context(steps, t=3, k=1, embed_fn=emb)
        self.assertEqual(len(r.retrieved_ts), 1)

    def test_chronological_assembly_order(self):
        steps = [
            _step(1, "t1", "x", "o1"),
            _step(2, "t2", "y", "o2"),
            _step(3, "t3", "z", "o3"),
        ]
        emb = deterministic_embed_fn(32)
        r = perfect_retrieve_context(steps, t=3, k=2, embed_fn=emb)
        self.assertEqual(list(r.retrieved_ts), sorted(r.retrieved_ts))
        parts = r.assembled_context.split("\n\n")
        self.assertEqual(len(parts), 2)


class TestQueryFallback(unittest.TestCase):
    def test_falls_back_to_thought_when_action_empty(self):
        steps = [_step(1, "only_thought", "", "obs")]
        emb = deterministic_embed_fn(16)
        r = perfect_retrieve_context(steps, t=1, k=1, embed_fn=emb)
        self.assertEqual(r.query_text, "only_thought")


class TestStepsToChunks(unittest.TestCase):
    def test_len_matches(self):
        steps = [_step(1, "a", "b", "c"), _step(2, "d", "e", "f")]
        ch = steps_to_chunks(steps)
        self.assertEqual(len(ch), 2)
        self.assertEqual(ch[0].t, 1)
        self.assertEqual(ch[1].t, 2)


if __name__ == "__main__":
    unittest.main()
