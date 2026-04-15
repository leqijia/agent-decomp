"""StateAct agent loop.

Variant of the ReAct agent where the model produces a structured state
summary S_t = (g, P_t, R_t, e_t, C, F_t, K_t) at every step before
deciding its action.  The state summary is self-generated (not oracle)
and is fed back into the prompt on subsequent steps so the agent
maintains an explicit running belief about task progress.

This is the "self_generated" condition in Experiment 3.  The oracle
external condition uses the same structured format but injects it from
an external oracle model (see harness/evaluator.py run_episode with
condition="oracle_external").

The loop is intentionally parallel to react_agent.py so both can be
compared fairly; differences are:
  1. Uses the p_stateact.json prompt (two-phase: state then action).
  2. Extracts and stores the <state>...</state> block each step.
  3. Feeds accumulated state summaries back into the prompt as context.
"""
from __future__ import annotations

import json
import os
import re
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

DEFAULT_PROMPT_PATH = Path(__file__).parent / "prompts" / "p_stateact.json"

_EARLY_STOP_K = 3
_API_RETRY_ATTEMPTS = 5
_API_RETRY_BASE_DELAY = 10


@dataclass
class StateActConfig:
    """Everything needed to run one StateAct episode."""
    config_file: str
    model: str
    prompt_path: Path = DEFAULT_PROMPT_PATH
    max_steps: int = 30
    temperature: float = 0.0
    max_obs_length: int = 4096
    observation_type: str = "accessibility_tree"
    action_set_tag: str = "id_accessibility_tree"
    agent_variant: str = "stateact"
    headless: bool = True
    sleep_after_execution: float = 2.0
    thinking: bool = True
    max_state_history: int = 5


@dataclass
class StateActResult:
    """In-memory mirror of a trajectories/SPEC.md record for StateAct."""
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


def _extract_state_json(response: str, open_tag: str, close_tag: str) -> dict | None:
    """Extract the structured state JSON from between <state>...</state> tags."""
    pattern = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
    match = re.search(pattern, response, re.DOTALL)
    if match is None:
        return None
    raw = match.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _format_state_for_context(state: dict, step_t: int) -> str:
    """Serialize one state summary into a compact line for prompt injection."""
    parts = [f"[Step {step_t} state]"]
    if state.get("g"):
        parts.append(f"Goal: {state['g']}")
    if state.get("P_t"):
        parts.append(f"Done: {', '.join(state['P_t'])}")
    if state.get("R_t"):
        parts.append(f"Remaining: {', '.join(state['R_t'])}")
    if state.get("e_t"):
        parts.append(f"Env: {state['e_t']}")
    if state.get("F_t"):
        failures = []
        for f in state["F_t"]:
            if isinstance(f, dict):
                failures.append(f"{f.get('action', '?')} (step {f.get('step', '?')}): {f.get('cause', '?')}")
            else:
                failures.append(str(f))
        parts.append(f"Failures: {'; '.join(failures)}")
    if state.get("K_t"):
        parts.append(f"Facts: {', '.join(str(k) for k in state['K_t'])}")
    return " | ".join(parts)


def _build_state_context(state_history: list[tuple[int, dict]], max_entries: int) -> str:
    """Build a context block from recent state summaries."""
    recent = state_history[-max_entries:]
    if not recent:
        return ""
    lines = [_format_state_for_context(state, t) for t, state in recent]
    return "STATE HISTORY:\n" + "\n".join(lines) + "\n"


def _split_thought_and_raw(response: str, answer_phrase: str) -> str:
    idx = response.find(answer_phrase)
    return response[:idx].strip() if idx >= 0 else response.strip()


def _load_task_config(config_file: str) -> dict[str, Any]:
    with open(config_file) as f:
        return json.load(f)


def _observation_text(obs: dict[str, Any]) -> str:
    text = obs.get("text")
    if not isinstance(text, str):
        raise RuntimeError(
            "Expected string text observation; got "
            f"{type(text).__name__}. Image-mode observation is not supported yet."
        )
    return text


def _is_repeat(a: str, b: str) -> bool:
    return a != "" and a == b


def run_stateact_episode(
    cfg: StateActConfig, *, out_path: str | Path | None = None
) -> StateActResult:
    """Run one task with the StateAct agent and return a SPEC.md-shaped result.

    Identical control flow to react_agent.run_episode except:
    - The prompt includes accumulated state summaries from prior steps.
    - Each step's response is parsed for a <state>...</state> JSON block.
    - The state summary is stored in each step dict as "state_summary".
    """
    template = PromptTemplate.load(cfg.prompt_path)
    task_config = _load_task_config(cfg.config_file)
    task_id = int(task_config["task_id"])
    intent = task_config["intent"]
    sites = task_config.get("sites", [])
    start_url = task_config.get("start_url", "")

    state_open = template.meta_data.get("state_open_tag", "<state>")
    state_close = template.meta_data.get("state_close_tag", "</state>")

    result = StateActResult(
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
    upstream_traj: list[Any] = []
    state_history: list[tuple[int, dict]] = []

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

            state_ctx = _build_state_context(state_history, cfg.max_state_history)
            augmented_obs = state_ctx + obs_truncated if state_ctx else obs_truncated

            messages = build_messages(
                template,
                objective=intent,
                observation=augmented_obs,
                url=step_url,
                previous_action=previous_action_str,
            )

            parse_error: str | None = None
            action_str = ""
            action_obj = create_none_action()

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
                    is_transient = any(code in err_str for code in (
                        "502", "503", "429", "UNAVAILABLE", "overloaded",
                        "high demand", "rate increased too quickly", "Alibaba",
                    ))
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
                            "", parse_error, prompt_tokens, completion_tokens,
                            latency_ms, None,
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

            state_json = _extract_state_json(raw_prediction, state_open, state_close)
            if state_json is not None:
                state_history.append((t, state_json))

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
                    action_str, parse_error, prompt_tokens, completion_tokens,
                    latency_ms, state_json,
                )
            )
            result.total_steps = t

            if parse_error is not None:
                recent_parse_failures += 1
                if recent_parse_failures >= _EARLY_STOP_K:
                    result.stop_reason = "parse_failures"
                    break
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
            except Exception as e:
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
        except Exception:
            result.eval_score = None
            result.success = None

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
    state_summary: dict | None,
) -> dict[str, Any]:
    d = {
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
    if state_summary is not None:
        d["state_summary"] = state_summary
    return d
