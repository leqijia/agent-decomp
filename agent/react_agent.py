"""ReAct agent loop.

Single-episode runner. Given a WebArena task config file, this drives the
vendored ScriptBrowserEnv with a Chain-of-Thought / ReAct-style prompt,
emits a trajectory JSON conforming to trajectories/SPEC.md, and scores it
with the upstream evaluator_router.

This is the primary data source for the paper. Every Experiment 1 source
trajectory, every Experiment 2 length bin, and every baseline condition is
either a direct call to `run_episode` here or calls it through the research
harness in harness/SPEC.md.

Design notes:
  - The agent maintains two parallel trajectory representations. The upstream
    `Trajectory = list[StateInfo | Action]` is required by the scorer; our
    SPEC.md list of step dicts is what we persist to disk. They are built
    in lockstep and never diverge.
  - Early-stop rules mirror upstream run.py: max_steps, K consecutive parse
    failures, K consecutive identical actions. These map to the stop_reason
    enum in trajectories/SPEC.md.
  - Qwen3 `<think>` mode is off by default. Week 1 collection is cheaper
    and easier to audit without it. Turning it on is a template swap, not a
    code change — pass a different prompt JSON.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.prompt_constructor import (
    PromptTemplate,
    build_messages,
    extract_action,
    truncate_observation,
)
from llm.client import chat_completion
from llm.openrouter import OpenRouterError
from webarena.browser_env import (
    ActionParsingError,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    create_id_based_action,
    create_none_action,
    create_stop_action,
)
from webarena.evaluation_harness.evaluators import evaluator_router

load_dotenv()

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "p_cot_id_actree_2s.json"

# Consecutive-failure and consecutive-repeat thresholds, matching upstream
# run.py defaults. These drive the parse_failures and repeating_action
# stop_reason values in trajectories/SPEC.md.
_EARLY_STOP_K = 3
_API_RETRY_ATTEMPTS = 5
_API_RETRY_BASE_DELAY = 10  # seconds; doubles each attempt (10, 20, 40, 80, 160)


@dataclass
class EpisodeConfig:
    """Everything needed to run one task and emit a SPEC.md-conformant file."""
    config_file: str                 # WebArena task JSON path
    model: str                       # OpenRouter slug, e.g. qwen/qwen3.5-27b
    prompt_path: Path = DEFAULT_PROMPT_PATH
    max_steps: int = 30
    temperature: float = 0.0
    max_obs_length: int = 4096
    observation_type: str = "accessibility_tree"
    action_set_tag: str = "id_accessibility_tree"
    agent_variant: str = "react"
    headless: bool = True
    sleep_after_execution: float = 2.0  # WebArena hard-sleep for page settling
    # Qwen3 reasoning mode. When True, passes {"reasoning": {"enabled": True}}
    # to OpenRouter so the model runs internal reasoning before its answer.
    thinking: bool = True


@dataclass
class EpisodeResult:
    """In-memory mirror of a trajectories/SPEC.md record."""
    task_id: int
    config_file: str
    sites: list[str]
    intent: str
    start_url: str
    model: str
    agent_variant: str
    observation_type: str
    action_set_tag: str
    max_steps: int
    temperature: float
    max_obs_length: int
    started_at: str
    ended_at: str
    total_steps: int
    stop_reason: str
    success: bool | None
    eval_score: float | None
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "config_file": self.config_file,
            "sites": self.sites,
            "intent": self.intent,
            "start_url": self.start_url,
            "model": self.model,
            "agent_variant": self.agent_variant,
            "observation_type": self.observation_type,
            "action_set_tag": self.action_set_tag,
            "max_steps": self.max_steps,
            "temperature": self.temperature,
            "max_obs_length": self.max_obs_length,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_steps": self.total_steps,
            "stop_reason": self.stop_reason,
            "success": self.success,
            "eval_score": self.eval_score,
            "steps": self.steps,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_thought_and_raw(response: str, answer_phrase: str) -> str:
    """Return the thought portion of a CoT response.

    The CoT template instructs the model to produce reasoning followed by
    "In summary, the next action I will perform is ```<action>```". The
    thought is everything before that phrase. If the phrase is absent
    (e.g. on parse failure), the entire response is the thought.
    """
    idx = response.find(answer_phrase)
    return response[:idx].strip() if idx >= 0 else response.strip()


def _load_task_config(config_file: str) -> dict[str, Any]:
    with open(config_file) as f:
        return json.load(f)


def _observation_text(obs: dict[str, Any]) -> str:
    """Extract the text observation the agent is prompted with.

    Upstream's observation dict is keyed by modality; the text key is "text"
    for both html and accessibility_tree modes. Image mode returns a numpy
    array under "image" and is out of scope for the Week 1 text pipeline.
    """
    text = obs.get("text")
    if not isinstance(text, str):
        raise RuntimeError(
            "Expected string text observation; got "
            f"{type(text).__name__}. Image-mode observation is not supported yet."
        )
    return text


def _is_repeat(a: str, b: str) -> bool:
    """Upstream's same-action rule: exact string match on the action string.

    Distinct from upstream's "same typing action" special case, which is
    redundant here because identical type-action strings are already caught
    by exact match on the canonical action string.
    """
    return a != "" and a == b


def run_episode(cfg: EpisodeConfig, *, out_path: str | Path | None = None) -> EpisodeResult:
    """Run one task end-to-end and return its SPEC.md-shaped result.

    Opens a fresh ScriptBrowserEnv, resets to cfg.config_file, prompts the
    agent until one of the stop conditions fires, scores the resulting
    trajectory with evaluator_router, and (if out_path is provided) writes
    the JSON to disk. The returned EpisodeResult is the same record in
    memory — callers that batch episodes can stream it into a list instead
    of re-reading from disk.

    Never raises on expected failures (parse errors, repeat loops, env
    errors from a single action); those become stop_reason values. If
    something truly unexpected raises, the trajectory is persisted with
    stop_reason=crash and success=None so the run can still be inspected.
    """
    template = PromptTemplate.load(cfg.prompt_path)
    task_config = _load_task_config(cfg.config_file)
    task_id = int(task_config["task_id"])
    intent = task_config["intent"]
    sites = task_config.get("sites", [])
    start_url = task_config.get("start_url", "")

    result = EpisodeResult(
        task_id=task_id,
        config_file=cfg.config_file,
        sites=sites,
        intent=intent,
        start_url=start_url,
        model=cfg.model,
        agent_variant=cfg.agent_variant,
        observation_type=cfg.observation_type,
        action_set_tag=cfg.action_set_tag,
        max_steps=cfg.max_steps,
        temperature=cfg.temperature,
        max_obs_length=cfg.max_obs_length,
        started_at=_now_iso(),
        ended_at="",
        total_steps=0,
        stop_reason="max_steps",
        success=None,
        eval_score=None,
    )

    env = ScriptBrowserEnv(
        headless=cfg.headless,
        observation_type=cfg.observation_type,
        sleep_after_execution=cfg.sleep_after_execution,
    )
    upstream_traj: list[Any] = []  # list[StateInfo | Action], for the scorer

    try:
        obs, info = env.reset(options={"config_file": cfg.config_file})
        state_info: StateInfo = {"observation": obs, "info": info}
        upstream_traj.append(state_info)

        previous_action_str = "None"
        recent_parse_failures = 0
        recent_actions: list[str] = []

        for t in range(1, cfg.max_steps + 1):
            step_url = env.page.url
            obs_text = _observation_text(obs)
            obs_truncated = truncate_observation(obs_text, cfg.max_obs_length)

            messages = build_messages(
                template,
                objective=intent,
                observation=obs_truncated,
                url=step_url,
                previous_action=previous_action_str,
            )

            parse_error: str | None = None
            action_str = ""
            action_obj = create_none_action()

            # Retry transient API errors (503, 429, timeouts) up to 3 times
            # with exponential backoff. Permanent failures still crash.
            thinking_body = (
                {"reasoning": {"enabled": True}}
                if cfg.thinking else None
            )
            chat = None
            for attempt in range(_API_RETRY_ATTEMPTS):
                try:
                    chat = chat_completion(
                        messages,
                        model=cfg.model,
                        temperature=cfg.temperature,
                        extra_body=thinking_body,
                    )
                    break
                except OpenRouterError as e:
                    err_str = str(e)
                    is_transient = any(code in err_str for code in ("502", "503", "429", "UNAVAILABLE", "overloaded", "high demand", "rate increased too quickly", "Alibaba"))
                    if is_transient and attempt < _API_RETRY_ATTEMPTS - 1:
                        time.sleep(_API_RETRY_BASE_DELAY * (2 ** attempt))
                        continue
                    raw_prediction = f"[API_ERROR] {e}"
                    prompt_tokens = None
                    completion_tokens = None
                    latency_ms = 0
                    parse_error = "api_error"
                    result.stop_reason = "crash"
                    result.steps.append(
                        _make_step_dict(
                            t, step_url, obs_truncated, "", raw_prediction,
                            "", parse_error, prompt_tokens, completion_tokens, latency_ms,
                        )
                    )
                    result.total_steps = t
                    break
            if chat is None and result.stop_reason == "crash":
                break

            if chat is not None:
                raw_prediction = chat.content
                prompt_tokens = chat.prompt_tokens
                completion_tokens = chat.completion_tokens
                latency_ms = chat.latency_ms

            try:
                action_str = extract_action(template, raw_prediction)
                action_obj = create_id_based_action(action_str)
            except ActionParsingError as e:
                parse_error = str(e)
                action_str = ""
                action_obj = create_none_action()

            action_obj["raw_prediction"] = raw_prediction
            upstream_traj.append(action_obj)

            thought = _split_thought_and_raw(
                raw_prediction, template.meta_data["answer_phrase"]
            )

            result.steps.append(
                _make_step_dict(
                    t, step_url, obs_truncated, thought, raw_prediction,
                    action_str, parse_error, prompt_tokens, completion_tokens, latency_ms,
                )
            )
            result.total_steps = t

            # Early-stop rules (match upstream run.py)
            if parse_error is not None:
                recent_parse_failures += 1
                if recent_parse_failures >= _EARLY_STOP_K:
                    result.stop_reason = "parse_failures"
                    break
                # Parse failure means no env.step; loop to retry with same obs.
                previous_action_str = "None"
                continue
            recent_parse_failures = 0

            recent_actions.append(action_str)
            if (
                len(recent_actions) >= _EARLY_STOP_K
                and all(_is_repeat(recent_actions[-1], a) for a in recent_actions[-_EARLY_STOP_K:])
            ):
                result.stop_reason = "repeating_action"
                break

            if action_obj["action_type"] == ActionTypes.STOP:
                result.stop_reason = "agent_stop"
                break

            try:
                obs, _reward, terminated, _truncated, info = env.step(action_obj)
            except Exception as e:  # env-level failure on one action
                parse_error = f"env_step_error: {e}"
                result.stop_reason = "crash"
                break

            state_info = {"observation": obs, "info": info}
            upstream_traj.append(state_info)
            previous_action_str = action_str

            if terminated:
                result.stop_reason = "env_terminated"
                break
        else:
            result.stop_reason = "max_steps"

        # Scoring requires a trajectory ending in an Action. If we broke out
        # on max_steps or env_terminated with a StateInfo at the tail, append
        # a synthetic stop action so the evaluator can read the last state.
        if upstream_traj and not isinstance(upstream_traj[-1], dict) or (
            upstream_traj and "action_type" not in upstream_traj[-1]
        ):
            upstream_traj.append(create_stop_action(""))

        try:
            scorer = evaluator_router(cfg.config_file)
            score = scorer(
                upstream_traj,
                cfg.config_file,
                env.page,
                env.get_page_client(env.page),
            )
            result.eval_score = float(score)
            result.success = score == 1.0
        except Exception as e:
            # Evaluator blew up: persist the run but mark scoring as unknown.
            result.eval_score = None
            result.success = None
            result.steps.append({"evaluator_error": str(e), "traceback": traceback.format_exc()})

    except Exception as e:
        result.stop_reason = "crash"
        result.success = None
        result.eval_score = None
        result.steps.append({"crash": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            env.close()
        except Exception:
            pass

    result.ended_at = _now_iso()
    print(f"  => stop={result.stop_reason} | steps={result.total_steps} | success={result.success} | score={result.eval_score}")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return result


def _make_step_dict(
    t: int,
    url: str,
    observation: str,
    thought: str,
    raw_prediction: str,
    action: str,
    parse_error: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    latency_ms: int,
) -> dict[str, Any]:
    """Build one step record in the exact shape trajectories/SPEC.md requires."""
    return {
        "t": t,
        "url": url,
        "observation": observation,
        "thought": thought,
        "raw_prediction": raw_prediction,
        "action": action,
        "parse_error": parse_error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_ms": latency_ms,
    }
