"""Tests for trajectory baselines (run: python -m unittest baselines.test_trajectory from oracle/)."""

import unittest

from baselines.trajectory import (
    compose_mask_then_window,
    compose_window_then_mask,
    count_tokens,
    format_step_line,
    mask_observations,
    serialize_trajectory,
    sliding_window_truncate,
)


def _step(t: int, thought: str, action: str, observation: str) -> dict:
    return {"t": t, "thought": thought, "action": action, "observation": observation}


class TestSerialize(unittest.TestCase):
    def test_format_matches_generate_oracle(self):
        s = _step(1, "a", "b()", "obs")
        self.assertEqual(
            format_step_line(s),
            "Step 1: thought=a | action=b() | observation=obs",
        )

    def test_serialize_empty(self):
        self.assertEqual(serialize_trajectory([]), "")

    def test_serialize_joins_newlines(self):
        steps = [_step(1, "x", "y", "z"), _step(2, "a", "b", "c")]
        self.assertEqual(
            serialize_trajectory(steps),
            format_step_line(steps[0]) + "\n" + format_step_line(steps[1]),
        )


class TestMaskObservations(unittest.TestCase):
    def test_preserves_thought_and_action(self):
        steps = [_step(1, "think", "act()", "huge" * 1000)]
        m = mask_observations(steps)
        self.assertEqual(m[0]["thought"], "think")
        self.assertEqual(m[0]["action"], "act()")
        self.assertEqual(m[0]["observation"], "[OBSERVATION_REDACTED]")

    def test_does_not_mutate_input(self):
        steps = [_step(1, "t", "a", "orig")]
        mask_observations(steps)
        self.assertEqual(steps[0]["observation"], "orig")

    def test_step_index_optional(self):
        steps = [_step(5, "t", "a", "o")]
        m = mask_observations(steps, step_index_in_placeholder=True)
        self.assertIn("step=5", m[0]["observation"])

    def test_keep_last_preserves_final_observation(self):
        steps = [
            _step(1, "a", "a", "first"),
            _step(2, "b", "b", "last_real"),
        ]
        m = mask_observations(steps, mask_mode="keep_last")
        self.assertEqual(m[0]["observation"], "[OBSERVATION_REDACTED]")
        self.assertEqual(m[1]["observation"], "last_real")


class TestCompose(unittest.TestCase):
    def test_mask_then_window_returns_steps_and_result(self):
        steps = [_step(1, "a", "a", "x"), _step(2, "b", "b", "y")]
        out, res = compose_mask_then_window(steps, max_tokens=10_000)
        self.assertEqual(len(out), len(res.steps))
        self.assertTrue(all(s["observation"] == "[OBSERVATION_REDACTED]" for s in out))

    def test_compose_orders_can_differ(self):
        steps = [
            _step(1, "a", "a", "obs1"),
            _step(2, "b", "b", "obs2"),
        ]
        m_w, _ = compose_mask_then_window(steps, max_tokens=500)
        w_m, _ = compose_window_then_mask(steps, max_tokens=500)
        # Same inputs; if both keep 2 steps, masking order still yields same obs text.
        self.assertEqual(m_w[0]["observation"], "[OBSERVATION_REDACTED]")
        self.assertEqual(w_m[0]["observation"], "[OBSERVATION_REDACTED]")


class TestSlidingWindow(unittest.TestCase):
    def test_no_drop_when_under_budget(self):
        steps = [_step(1, "a", "b", "c")]
        r = sliding_window_truncate(steps, max_tokens=10_000)
        self.assertEqual(r.steps, steps)
        self.assertEqual(r.dropped_prefix_steps, 0)
        self.assertFalse(r.last_step_exceeds_budget)

    def test_drops_earliest_steps_to_fit(self):
        steps = [
            _step(1, "a", "a", "a"),
            _step(2, "b", "b", "b"),
            _step(3, "c", "c", "c"),
        ]
        full_tok = count_tokens(serialize_trajectory(steps))
        mid = count_tokens(serialize_trajectory(steps[1:]))
        self.assertLess(mid, full_tok)
        r = sliding_window_truncate(steps, max_tokens=mid)
        self.assertEqual(r.steps, steps[1:])
        self.assertEqual(r.dropped_prefix_steps, 1)

    def test_last_step_only_when_impossible(self):
        long_obs = "x" * 50_000
        steps = [_step(1, "t", "a()", long_obs)]
        r = sliding_window_truncate(steps, max_tokens=10)
        self.assertEqual(len(r.steps), 1)
        self.assertTrue(r.last_step_exceeds_budget)


if __name__ == "__main__":
    unittest.main()
