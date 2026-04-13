"""Tests for ACON-style compression and guideline loop (no API keys required)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baselines.acon.compress import compress_trajectory
from baselines.acon.guideline_update import optimize_guidelines_round, propose_guideline_update
from baselines.acon.io import (
    load_pair_records_jsonl,
    resolve_full_trajectory_text,
    save_guideline_state,
    load_guideline_state,
)
from baselines.acon.parsing import extract_compressed_context, extract_revised_guideline
from baselines.acon.templating import build_compress_user_message, build_guideline_update_user_message
from baselines.acon.types import GuidelineState, PairRecord


def _step(t: int, thought: str, action: str, observation: str) -> dict:
    return {"t": t, "thought": thought, "action": action, "observation": observation}


class TestParsing(unittest.TestCase):
    def test_extract_compressed(self):
        raw = "Preamble\n### COMPRESSED_CONTEXT\nhello\nworld"
        self.assertEqual(extract_compressed_context(raw), "hello\nworld")

    def test_extract_guideline(self):
        raw = "### REVISED_GUIDELINE\nKeep all URLs."
        self.assertEqual(extract_revised_guideline(raw), "Keep all URLs.")


class TestTemplatingGolden(unittest.TestCase):
    def test_compress_template_contains_placeholders_resolved(self):
        msg = build_compress_user_message(
            intent="Do X",
            guideline="Keep Y",
            trajectory_text="Step 1: ...",
        )
        self.assertIn("Do X", msg)
        self.assertIn("Keep Y", msg)
        self.assertIn("Step 1:", msg)
        self.assertIn("### COMPRESSED_CONTEXT", msg)

    def test_guideline_update_template(self):
        msg = build_guideline_update_user_message(
            guideline="G",
            intent="I",
            full_trajectory_or_summary="FULL",
            compressed_context_used="COMP",
            failure_signal="F",
        )
        self.assertIn("G", msg)
        self.assertIn("FULL", msg)
        self.assertIn("COMP", msg)
        self.assertIn("F", msg)
        self.assertIn("### REVISED_GUIDELINE", msg)


class TestCompressStub(unittest.TestCase):
    def test_compress_uses_stub_llm(self):
        def stub(msg: str) -> str:
            return "### COMPRESSED_CONTEXT\nSHORT"

        out = compress_trajectory(
            "task",
            [_step(1, "a", "b", "c")],
            "guideline",
            llm=stub,
        )
        self.assertEqual(out.compressed_text, "SHORT")
        self.assertEqual(out.model, "stub")
        self.assertGreater(out.input_tokens, 0)


class TestGuidelineUpdate(unittest.TestCase):
    def test_skip_when_not_qualifying_pair(self):
        pair = PairRecord(
            task_id=1,
            intent="x",
            full_success=False,
            compressed_success=False,
            full_trajectory_text="full",
            compressed_context_used="comp",
        )
        state = GuidelineState(text="g0", version=0)
        new_text, raw = propose_guideline_update(pair, state, llm=lambda _: "BAD")
        self.assertEqual(new_text, "g0")
        self.assertEqual(raw, "")

    def test_propose_updates_with_stub(self):
        pair = PairRecord(
            task_id=1,
            intent="open repo",
            full_success=True,
            compressed_success=False,
            full_trajectory_text="Step 1: thought=t | action=a | observation=o",
            compressed_context_used="too short",
            failure_signal="eval failed",
        )
        state = GuidelineState(text="old", version=0)

        def stub(_: str) -> str:
            return "### REVISED_GUIDELINE\nPreserve element ids in observations."

        new_text, raw = propose_guideline_update(pair, state, llm=stub)
        self.assertEqual(new_text, "Preserve element ids in observations.")
        self.assertIn("REVISED_GUIDELINE", raw)

    def test_optimize_round_sequential(self):
        p1 = PairRecord(
            task_id=1,
            intent="a",
            full_success=True,
            compressed_success=False,
            full_trajectory_text="t1",
            compressed_context_used="c1",
        )
        p2 = PairRecord(
            task_id=2,
            intent="b",
            full_success=True,
            compressed_success=False,
            full_trajectory_text="t2",
            compressed_context_used="c2",
        )
        calls: list[str] = []

        def stub(msg: str) -> str:
            calls.append(msg)
            if "t1" in msg and "c1" in msg:
                return "### REVISED_GUIDELINE\nafter_first"
            return "### REVISED_GUIDELINE\nafter_second"

        s0 = GuidelineState(text="start", version=0)
        s1 = optimize_guidelines_round([p1, p2], s0, llm=stub)
        self.assertEqual(s1.text, "after_second")
        self.assertEqual(s1.version, 2)
        self.assertEqual(len(calls), 2)
        self.assertIn("after_first", calls[1])


class TestIoJsonl(unittest.TestCase):
    def test_load_pair_records_and_resolve_path(self):
        traj = {
            "task_id": 99,
            "steps": [_step(1, "x", "y", "z")],
        }
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "t.json"
            tp.write_text(json.dumps(traj), encoding="utf-8")
            jp = Path(td) / "pairs.jsonl"
            jp.write_text(
                json.dumps(
                    {
                        "task_id": 99,
                        "intent": "hi",
                        "full_success": True,
                        "compressed_success": False,
                        "path_full_traj": str(tp),
                        "compressed_context_used": "x",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            pairs = load_pair_records_jsonl(jp)
            self.assertEqual(len(pairs), 1)
            text = resolve_full_trajectory_text(pairs[0])
            self.assertIn("Step 1:", text)

    def test_guideline_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "g.json"
            g = GuidelineState(text="hello", version=3, meta={"k": "v"})
            save_guideline_state(p, g)
            g2 = load_guideline_state(p)
            self.assertEqual(g2.text, "hello")
            self.assertEqual(g2.version, 3)
            self.assertEqual(g2.meta["k"], "v")


if __name__ == "__main__":
    unittest.main()
