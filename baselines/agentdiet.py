"""AgentDiet-style heuristic filtering baseline.

Implements the core idea from the AgentDiet paper: remove expired and
redundant items from the trajectory context to reduce token usage without
losing decision-relevant information.

Three filtering heuristics:

1. **Expired action removal**: Navigation actions (goto, go_back,
   go_forward, scroll, new_tab, tab_focus, close_tab) whose effects are
   fully superseded by later actions are dropped.  For example, if the
   agent navigated to page A then page B, the goto-A step is expired
   because the agent is no longer on page A.

2. **Redundant observation dedup**: If two consecutive steps have
   observations whose Jaccard token overlap exceeds a threshold, the
   earlier observation is replaced with a short pointer
   "[see step {t} observation]" referencing the later one.

3. **Failed-action compression**: Consecutive failed actions (parse_error
   is not None) are collapsed into a single summary line rather than
   keeping each failed step verbatim.

These are pure trajectory transforms that take a step list and return a
filtered step list.  They compose with other baselines (e.g. sliding
window) and plug into the harness as a ContextPolicy.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import tiktoken

_NAVIGATION_ACTIONS = frozenset({
    "goto", "go_back", "go_forward", "scroll", "new_tab",
    "tab_focus", "close_tab",
})

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _action_verb(action_str: str) -> str:
    """Extract the first word (verb) from a canonical action string."""
    return action_str.split()[0].lower() if action_str else ""


def _tokenize_set(text: str) -> set[int]:
    return set(_ENCODING.encode(text))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def remove_expired_navigations(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop navigation steps whose effect is superseded by a later step.

    A navigation step at index i is "expired" if there exists a later
    navigation step at index j > i.  The idea: the agent has moved on,
    so the old navigation adds no decision-relevant context.

    Non-navigation steps (click, type, hover, press, stop) are always kept.
    The last navigation step is always kept.
    """
    if len(steps) <= 1:
        return deepcopy(steps)

    nav_indices = [
        i for i, s in enumerate(steps)
        if _action_verb(s.get("action", "")) in _NAVIGATION_ACTIONS
    ]

    if len(nav_indices) <= 1:
        return deepcopy(steps)

    expired = set(nav_indices[:-1])
    return [deepcopy(s) for i, s in enumerate(steps) if i not in expired]


def dedup_observations(
    steps: list[dict[str, Any]],
    jaccard_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Replace near-duplicate observations with back-references.

    Compares each step's observation to the next step's observation.
    If Jaccard token overlap >= threshold, the earlier observation is
    replaced with a pointer string.  The last step is never modified.
    """
    if len(steps) <= 1:
        return deepcopy(steps)

    out = []
    for i, step in enumerate(steps):
        row = deepcopy(step)
        if i < len(steps) - 1:
            obs_cur = step.get("observation", "")
            obs_next = steps[i + 1].get("observation", "")
            if obs_cur and obs_next:
                toks_cur = _tokenize_set(obs_cur)
                toks_next = _tokenize_set(obs_next)
                if _jaccard(toks_cur, toks_next) >= jaccard_threshold:
                    row["observation"] = f"[similar to step {steps[i+1]['t']} observation]"
        out.append(row)
    return out


def compress_failed_runs(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive failed actions into summary lines.

    A "failed" step is one where parse_error is not None.  Runs of 2+
    consecutive failures are replaced with a single synthetic step
    summarizing the run.
    """
    if not steps:
        return []

    out: list[dict[str, Any]] = []
    i = 0
    while i < len(steps):
        if steps[i].get("parse_error") is not None:
            run_start = i
            while i < len(steps) and steps[i].get("parse_error") is not None:
                i += 1
            run_len = i - run_start
            if run_len >= 2:
                first_t = steps[run_start]["t"]
                last_t = steps[i - 1]["t"]
                out.append({
                    "t": first_t,
                    "url": steps[run_start].get("url", ""),
                    "observation": f"[{run_len} consecutive parse failures, steps {first_t}-{last_t}]",
                    "thought": f"Failed to parse actions for {run_len} consecutive steps.",
                    "raw_prediction": "",
                    "action": "",
                    "parse_error": f"collapsed_{run_len}_failures",
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "latency_ms": 0,
                })
            else:
                out.append(deepcopy(steps[run_start]))
        else:
            out.append(deepcopy(steps[i]))
            i += 1
    return out


def agentdiet_filter(
    steps: list[dict[str, Any]],
    *,
    remove_nav: bool = True,
    dedup_obs: bool = True,
    compress_failures: bool = True,
    jaccard_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """Apply the full AgentDiet filtering pipeline.

    Applies heuristics in order: expired navigation removal, observation
    dedup, failed-action compression.  Each can be toggled independently
    for ablation.
    """
    result = steps
    if remove_nav:
        result = remove_expired_navigations(result)
    if dedup_obs:
        result = dedup_observations(result, jaccard_threshold=jaccard_threshold)
    if compress_failures:
        result = compress_failed_runs(result)
    return result
