"""Evaluation harness — single entry point for every experimental condition.

Implements the three functions from harness/SPEC.md:

  run_episode       — fresh rollout under a condition
  run_intervention  — Exp 1 replay-and-intervene
  evaluate_batch    — run_episode across a list of tasks
  compute_metrics   — aggregate results

Plus the ACON compression helper from Marlin's earlier stub.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from oracle.generate_oracle import build_prompt, call_oracle, get_live_dom, save_output

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from baselines.acon import compress_trajectory
from baselines.agentdiet import agentdiet_filter
from baselines.trajectory import (
    mask_observations,
    serialize_trajectory,
    sliding_window_truncate,
)
from harness.conditions import CONDITION_REGISTRY, VALID_CONDITIONS
from harness.metrics import LENGTH_BINS, _length_bin
from llm.config import AGENT_MODEL

load_dotenv()


# ---------------------------------------------------------------------------
# ACON stub (preserved from Marlin's original)
# ---------------------------------------------------------------------------

def _stub_llm(msg: str) -> str:
    return "### COMPRESSED_CONTEXT\n" + msg[:500]


def run_acon_compress(
    intent: str,
    steps: list[dict],
    guideline: str = "",
    *,
    llm=None,
    model: str = "",
) -> dict:
    """Compress a trajectory via the ACON pipeline.

    By default uses the live OpenRouter LLM (`baselines.acon.openrouter_generate`)
    with `llm.config.ACON_MODEL`. When `OPENROUTER_API_KEY` is unset the call
    transparently falls back to `_stub_llm`, keeping the offline unit test in
    `[harness/test_harness.py](harness/test_harness.py)` self-contained
    (it asserts `result["model"] == "stub"`, which `compress_trajectory`
    sets when the `llm` argument is a non-openrouter callable).

    Pass `llm=` to override (tests, alternate providers, deterministic stubs).
    """
    if llm is None and not os.environ.get("OPENROUTER_API_KEY"):
        llm = _stub_llm
    result = compress_trajectory(intent, steps, guideline, llm=llm, model=model)
    return {
        "compressed_text": result.compressed_text,
        "input_tokens": result.input_tokens,
        "model": result.model,
    }


# ---------------------------------------------------------------------------
# Context policies
# ---------------------------------------------------------------------------

def _identity_policy(
    steps_so_far: list[dict], current_observation: str, intent: str
) -> str:
    """Raw context — full step history serialized as-is."""
    return serialize_trajectory(steps_so_far)


def _sliding_window_policy(
    steps_so_far: list[dict],
    current_observation: str,
    intent: str,
    *,
    window_size: int = 4096,
) -> str:
    result = sliding_window_truncate(steps_so_far, window_size)
    return serialize_trajectory(result.steps)


def _observation_masking_policy(
    steps_so_far: list[dict], current_observation: str, intent: str
) -> str:
    masked = mask_observations(steps_so_far, mask_mode="keep_last")
    return serialize_trajectory(masked)


def _agentdiet_policy(
    steps_so_far: list[dict], current_observation: str, intent: str
) -> str:
    filtered = agentdiet_filter(steps_so_far)
    return serialize_trajectory(filtered)


def _acon_policy(
    steps_so_far: list[dict], current_observation: str, intent: str
) -> str:
    result = run_acon_compress(intent, steps_so_far)
    return result["compressed_text"]


def _env_only_policy(
    steps_so_far: list[dict], current_observation: str, intent: str
) -> str:
    """Minimal context: only the goal and current observation (no history)."""
    return ""


def _oracle_external_policy(
    steps_so_far: list[dict],
    current_observation: str,
    intent: str,
    *,
    oracle_states: dict[int, str] | None = None,
    regen_every_k: int = 3,
    current_url: str | None = None,
    task_id: int | str | None = None,
) -> str:
    """Inject pre-computed oracle state every k steps.

    oracle_states maps step number -> serialized oracle state string.
    Between oracle injections, falls back to identity (full history).
    """
    if oracle_states is None:
        return serialize_trajectory(steps_so_far)
    current_t = len(steps_so_far)

    if current_t % regen_every_k == 0:
        live_dom = get_live_dom(current_url)
        prompt_version = "v3"
        prompt = build_prompt(intent, steps_so_far, live_dom, current_t, version=prompt_version)
        result = call_oracle(prompt, use_stub=False)
        save_output(
            trajectory_id=str(task_id) if task_id is not None else "unknown",
            step=current_t,
            prompt_version=prompt_version,
            response_dict=result,
            output_dir=os.path.join(os.path.dirname(__file__), "tmp_oracle_outputs"),
        )
        oracle_states[current_t] = result["content"]

    latest_oracle_t = None
    for t in sorted(oracle_states.keys(), reverse=True):
        if t <= current_t:
            latest_oracle_t = t 
            break
    if latest_oracle_t is not None:
        return oracle_states[latest_oracle_t]
    return serialize_trajectory(steps_so_far)


def _perfect_retrieval_policy(
    steps_so_far: list[dict],
    current_observation: str,
    intent: str,
    *,
    k: int = 3,
) -> str:
    """Live-time perfect-retrieval baseline.

    NOTE: this departs from baselines/perfect_retrieval.py's strict-oracle
    semantics, which use the future logged action at step t as the retrieval
    query (an upper bound that requires post-hoc replay). At live decision
    time we do not have a future action, so we query with
    `intent + current_observation`. The retrieval mechanism (cosine over
    past-step chunks) is unchanged; only the query construction differs.

    The embedding function is `deterministic_embed_fn` -- a SHA256-seeded
    pseudo-random projection. It is reproducible and dependency-light but not
    semantic; swap for `text-embedding-3-small` (or sentence-transformers) for
    paper-grade retrieval quality. Contract is unchanged: any
    `Callable[[list[str]], np.ndarray]` works.

    Falls back to `serialize_trajectory(steps_so_far)` when there are fewer
    than 2 past steps to retrieve from -- nothing meaningful to rank.
    """
    if len(steps_so_far) < 2:
        return serialize_trajectory(steps_so_far)

    import numpy as np
    from baselines.perfect_retrieval import (
        _normalize_rows,
        chunk_text_for_step,
        deterministic_embed_fn,
    )

    embed_fn = deterministic_embed_fn()
    corpus = [chunk_text_for_step(s) for s in steps_so_far]
    query = f"{intent}\n{current_observation}"

    q_vec = _normalize_rows(embed_fn([query]))[0]
    c_mat = _normalize_rows(embed_fn(corpus))
    sims = c_mat @ q_vec

    top_idx = np.argsort(-sims)[: min(k, len(corpus))]
    selected = sorted((int(i), corpus[int(i)]) for i in top_idx)
    return "\n\n".join(text for _, text in selected)


def _build_context_policy(
    condition: str,
    *,
    window_size: int | None = None,
    oracle_regen_every_k: int | None = None,
    oracle_states: dict[int, str] | None = None,
    perfect_retrieval_k: int | None = None,
):
    """Return a callable context policy for the given condition."""
    _, policy_name = CONDITION_REGISTRY[condition]

    if policy_name == "identity":
        return _identity_policy
    elif policy_name == "sliding_window":
        ws = window_size or 4096
        def _sw(steps, obs, intent):
            return _sliding_window_policy(steps, obs, intent, window_size=ws)
        return _sw
    elif policy_name == "observation_masking":
        return _observation_masking_policy
    elif policy_name == "agentdiet":
        return _agentdiet_policy
    elif policy_name == "acon":
        return _acon_policy
    elif policy_name == "env_only":
        return _env_only_policy
    elif policy_name == "oracle_external":
        k = oracle_regen_every_k or 3
        def _oe(steps, obs, intent, *, current_url=None, task_id = None):
            return _oracle_external_policy(
                steps, obs, intent,
                oracle_states=oracle_states,
                regen_every_k=k,
                current_url=current_url,
                task_id=task_id,
            )
        return _oe
    elif policy_name == "perfect_retrieval":
        pr_k = perfect_retrieval_k if perfect_retrieval_k is not None else 3
        def _pr(steps, obs, intent):
            return _perfect_retrieval_policy(steps, obs, intent, k=pr_k)
        return _pr
    else:
        raise ValueError(f"Unknown policy: {policy_name}")


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------

def run_episode(
    condition: str,
    config_file: str,
    *,
    model: str = AGENT_MODEL,
    max_steps: int = 75,
    max_obs_length: int = 4096,
    temperature: float = 0.0,
    window_size: int | None = None,
    oracle_regen_every_k: int | None = None,
    oracle_states: dict[int, str] | None = None,
    perfect_retrieval_k: int | None = None,
    out_path: str | None = None,
    thinking: bool = True,
) -> dict:
    """Run a single episode under a given experimental condition.

    Dispatches to the correct agent variant (react or stateact) and applies
    the condition's context policy to transform the step history before
    each prompt.

    Returns a dict conforming to trajectories/SPEC.md.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}. Valid: {VALID_CONDITIONS}")

    agent_variant, policy_name = CONDITION_REGISTRY[condition]
    context_policy = _build_context_policy(
        condition,
        window_size=window_size,
        oracle_regen_every_k=oracle_regen_every_k,
        oracle_states=oracle_states,
        perfect_retrieval_k=perfect_retrieval_k,
    )

    if agent_variant == "stateact":
        from agent.stateact_agent import StateActConfig, run_stateact_episode
        cfg = StateActConfig(
            config_file=config_file,
            model=model,
            max_steps=max_steps,
            max_obs_length=max_obs_length,
            temperature=temperature,
            thinking=thinking,
        )
        result = run_stateact_episode(cfg, out_path=out_path)
        return result.to_dict()

    # react agent with context policy applied
    from agent.react_agent import EpisodeConfig, run_episode as _run_react_episode
    cfg = EpisodeConfig(
        config_file=config_file,
        model=model,
        max_steps=max_steps,
        max_obs_length=max_obs_length,
        temperature=temperature,
        thinking=thinking,
    )

    if policy_name == "identity":
        result = _run_react_episode(cfg, out_path=out_path)
        return result.to_dict()

    # For non-identity policies, we run the react loop manually with
    # the context policy injected.  This is necessary because the react
    # agent loop does not natively support context policies -- it always
    # uses the full observation.  We import the internals needed.
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
    import time
    from datetime import datetime, timezone

    template = PromptTemplate.load(cfg.prompt_path)
    with open(config_file) as f:
        task_config = json.load(f)
    task_id = int(task_config["task_id"])
    intent = task_config["intent"]
    sites = task_config.get("sites", [])
    start_url = task_config.get("start_url", "")

    now_iso = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result_dict: dict[str, Any] = {
        "task_id": task_id,
        "config_file": config_file,
        "sites": sites,
        "intent": intent,
        "start_url": start_url,
        "model": model,
        "agent_variant": "react",
        "condition": condition,
        "observation_type": cfg.observation_type,
        "action_set_tag": cfg.action_set_tag,
        "max_steps": max_steps,
        "temperature": temperature,
        "max_obs_length": max_obs_length,
        "started_at": now_iso(),
        "ended_at": "",
        "total_steps": 0,
        "stop_reason": "max_steps",
        "success": None,
        "eval_score": None,
        "steps": [],
    }

    env = ScriptBrowserEnv(
        headless=cfg.headless,
        observation_type=cfg.observation_type,
        sleep_after_execution=cfg.sleep_after_execution,
    )
    upstream_traj: list[Any] = []
    _EARLY_STOP_K = 3
    _API_RETRY_ATTEMPTS = 5
    _API_RETRY_BASE_DELAY = 10

    try:
        obs, info = env.reset(options={"config_file": config_file})
        upstream_traj.append({"observation": obs, "info": info})

        previous_action_str = "None"
        recent_parse_failures = 0
        recent_actions: list[str] = []
        steps_so_far: list[dict] = []

        for t in range(1, max_steps + 1):
            step_url = env.page.url
            obs_text = obs.get("text", "")
            obs_truncated = truncate_observation(obs_text, max_obs_length)

            if policy_name == "oracle_external":
                ctx = context_policy(
                    steps_so_far,
                    obs_truncated,
                    intent,
                    current_url=step_url,
                    task_id=task_id
                )
            else:
                ctx = context_policy(steps_so_far, obs_truncated, intent)
            if ctx:
                augmented_obs = f"CONTEXT:\n{ctx}\n\nCURRENT OBSERVATION:\n{obs_truncated}"
            else:
                augmented_obs = obs_truncated

            messages = build_messages(
                template,
                objective=intent,
                observation=augmented_obs,
                url=step_url,
                previous_action=previous_action_str,
            )

            parse_error = None
            action_str = ""
            action_obj = create_none_action()

            thinking_body = (
                {"reasoning": {"enabled": True}} if thinking else None
            )
            chat = None
            for attempt in range(_API_RETRY_ATTEMPTS):
                try:
                    chat = chat_completion(
                        messages, model=model, temperature=temperature,
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
                    parse_error = "api_error"
                    result_dict["stop_reason"] = "crash"
                    step_dict = {
                        "t": t, "url": step_url, "observation": obs_truncated,
                        "thought": "", "raw_prediction": raw_prediction,
                        "action": "", "parse_error": parse_error,
                        "prompt_tokens": None, "completion_tokens": None,
                        "latency_ms": 0,
                    }
                    result_dict["steps"].append(step_dict)
                    steps_so_far.append(step_dict)
                    result_dict["total_steps"] = t
                    break

            if chat is None and result_dict["stop_reason"] == "crash":
                break

            if chat is not None:
                raw_prediction = chat.content
                prompt_tokens = chat.prompt_tokens
                completion_tokens = chat.completion_tokens
                latency_ms = chat.latency_ms

            try:
                action_str = extract_action(template, raw_prediction)
                action_obj = create_id_based_action(action_str)
            except ActionParsingError:
                parse_error = "parse_error"
                action_str = ""
                action_obj = create_none_action()

            action_obj["raw_prediction"] = raw_prediction
            upstream_traj.append(action_obj)

            answer_phrase = template.meta_data["answer_phrase"]
            idx = raw_prediction.find(answer_phrase)
            thought = raw_prediction[:idx].strip() if idx >= 0 else raw_prediction.strip()

            step_dict = {
                "t": t, "url": step_url, "observation": obs_truncated,
                "thought": thought, "raw_prediction": raw_prediction,
                "action": action_str, "parse_error": parse_error,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
            }
            result_dict["steps"].append(step_dict)
            steps_so_far.append(step_dict)
            result_dict["total_steps"] = t

            if parse_error is not None:
                recent_parse_failures += 1
                if recent_parse_failures >= _EARLY_STOP_K:
                    result_dict["stop_reason"] = "parse_failures"
                    break
                previous_action_str = "None"
                continue
            recent_parse_failures = 0

            recent_actions.append(action_str)
            if (
                len(recent_actions) >= _EARLY_STOP_K
                and all(
                    action_str != "" and action_str == a
                    for a in recent_actions[-_EARLY_STOP_K:]
                )
            ):
                result_dict["stop_reason"] = "repeating_action"
                break

            if action_obj["action_type"] == ActionTypes.STOP:
                result_dict["stop_reason"] = "agent_stop"
                break

            try:
                obs, _reward, terminated, _truncated, info = env.step(action_obj)
            except Exception as e:
                result_dict["stop_reason"] = "crash"
                break

            upstream_traj.append({"observation": obs, "info": info})
            previous_action_str = action_str

            if terminated:
                result_dict["stop_reason"] = "env_terminated"
                break
        else:
            result_dict["stop_reason"] = "max_steps"

        # Per SPEC: success/eval_score must be null on crash.
        if result_dict.get("stop_reason") != "crash":
            if upstream_traj and (
                not isinstance(upstream_traj[-1], dict)
                or "action_type" not in upstream_traj[-1]
            ):
                upstream_traj.append(create_stop_action(""))

            try:
                scorer = evaluator_router(config_file)
                score = scorer(upstream_traj, config_file, env.page, env.get_page_client(env.page))
                result_dict["eval_score"] = float(score)
                result_dict["success"] = score == 1.0
            except Exception:
                result_dict["eval_score"] = None
                result_dict["success"] = None

    except Exception as e:
        result_dict["stop_reason"] = "crash"
        result_dict["success"] = None
        result_dict["eval_score"] = None
        result_dict["steps"].append({"crash": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            env.close()
        except Exception:
            pass

    result_dict["ended_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"  => condition={condition} stop={result_dict['stop_reason']} | steps={result_dict['total_steps']} | success={result_dict['success']}")

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result_dict, f, indent=2)

    return result_dict


# ---------------------------------------------------------------------------
# run_intervention (Experiment 1)
# ---------------------------------------------------------------------------

def run_intervention(
    trajectory_path: str,
    t_star: int,
    replacement_context: str,
    *,
    model: str = AGENT_MODEL,
    max_steps: int = 75,
    max_obs_length: int = 4096,
    temperature: float = 0.0,
    out_path: str | None = None,
    thinking: bool = True,
    env_only: bool = False,
) -> dict:
    """Replay a stored trajectory to t_star, then resume with replaced context.

    Semantics (from harness/SPEC.md):
    1. Load stored trajectory, open a fresh env from its config_file.
    2. Replay steps 1..t_star-1 by feeding stored actions through env.step.
    3. At t_star, resume the agent with replacement_context (oracle state)
       substituted for the step-log serialization, OR with the raw context
       if env_only=True (environment-only control).
    4. Continue the normal loop from t_star until stop/max_steps.
    5. Score and emit a SPEC.md-conformant trajectory.

    env_only=True is the environment-only control: env is restored to the
    correct state at t_star but the agent keeps its raw trajectory context.
    """
    import time
    from datetime import datetime, timezone

    from agent.prompt_constructor import (
        PromptTemplate,
        build_messages,
        extract_action,
        truncate_observation,
    )
    from agent.react_agent import DEFAULT_PROMPT_PATH
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

    with open(trajectory_path) as f:
        stored = json.load(f)

    config_file = stored["config_file"]
    task_id = stored["task_id"]
    intent = stored["intent"]
    sites = stored.get("sites", [])
    start_url = stored.get("start_url", "")
    stored_steps = stored["steps"]

    if t_star < 1 or t_star > len(stored_steps):
        raise ValueError(
            f"t_star={t_star} out of range for trajectory with {len(stored_steps)} steps"
        )

    template = PromptTemplate.load(DEFAULT_PROMPT_PATH)
    now_iso = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    condition_label = "env_only" if env_only else "oracle_external"
    result_dict: dict[str, Any] = {
        "task_id": task_id,
        "config_file": config_file,
        "sites": sites,
        "intent": intent,
        "start_url": start_url,
        "model": model,
        "agent_variant": condition_label,
        "observation_type": stored.get("observation_type", "accessibility_tree"),
        "action_set_tag": stored.get("action_set_tag", "id_accessibility_tree"),
        "max_steps": max_steps,
        "temperature": temperature,
        "max_obs_length": max_obs_length,
        "intervention": {
            "t_star": t_star,
            "source_trajectory": trajectory_path,
            "env_only": env_only,
        },
        "started_at": now_iso(),
        "ended_at": "",
        "total_steps": 0,
        "stop_reason": "max_steps",
        "success": None,
        "eval_score": None,
        "steps": [],
    }

    env = ScriptBrowserEnv(
        headless=True,
        observation_type="accessibility_tree",
        sleep_after_execution=2.0,
    )
    upstream_traj: list[Any] = []
    _EARLY_STOP_K = 3
    _API_RETRY_ATTEMPTS = 5
    _API_RETRY_BASE_DELAY = 10

    try:
        obs, info = env.reset(options={"config_file": config_file})
        upstream_traj.append({"observation": obs, "info": info})

        # Phase 1: Replay steps 1..t_star-1
        for replay_step in stored_steps[:t_star - 1]:
            stored_action = replay_step.get("action", "")
            if not stored_action:
                continue
            try:
                action_obj = create_id_based_action(stored_action)
                action_obj["raw_prediction"] = replay_step.get("raw_prediction", "")
                upstream_traj.append(action_obj)
                obs, _reward, terminated, _truncated, info = env.step(action_obj)
                upstream_traj.append({"observation": obs, "info": info})
            except (ActionParsingError, Exception) as e:
                result_dict["stop_reason"] = "replay_desync"
                result_dict["total_steps"] = replay_step["t"]
                result_dict["steps"].append({
                    "replay_desync": str(e),
                    "desync_at_step": replay_step["t"],
                })
                result_dict["ended_at"] = now_iso()
                if out_path:
                    p = Path(out_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "w") as f:
                        json.dump(result_dict, f, indent=2)
                try:
                    env.close()
                except Exception:
                    pass
                return result_dict

            if terminated:
                result_dict["stop_reason"] = "replay_desync"
                result_dict["ended_at"] = now_iso()
                if out_path:
                    p = Path(out_path)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with open(p, "w") as f:
                        json.dump(result_dict, f, indent=2)
                try:
                    env.close()
                except Exception:
                    pass
                return result_dict

        # Phase 2: Resume from t_star with intervention
        previous_action_str = "None"
        if t_star > 1 and stored_steps[t_star - 2].get("action"):
            previous_action_str = stored_steps[t_star - 2]["action"]

        recent_parse_failures = 0
        recent_actions: list[str] = []
        steps_collected: list[dict] = list(stored_steps[:t_star - 1])

        remaining_steps = max_steps - (t_star - 1)
        for t_offset in range(remaining_steps):
            t = t_star + t_offset
            step_url = env.page.url
            obs_text = obs.get("text", "") if isinstance(obs, dict) else ""
            obs_truncated = truncate_observation(obs_text, max_obs_length)

            if env_only:
                ctx = serialize_trajectory(steps_collected)
            else:
                if t_offset == 0:
                    ctx = replacement_context
                else:
                    ctx = serialize_trajectory(steps_collected)

            if ctx:
                augmented_obs = f"CONTEXT:\n{ctx}\n\nCURRENT OBSERVATION:\n{obs_truncated}"
            else:
                augmented_obs = obs_truncated

            messages = build_messages(
                template,
                objective=intent,
                observation=augmented_obs,
                url=step_url,
                previous_action=previous_action_str,
            )

            parse_error = None
            action_str = ""
            action_obj = create_none_action()
            thinking_body = {"reasoning": {"enabled": True}} if thinking else None

            chat = None
            for attempt in range(_API_RETRY_ATTEMPTS):
                try:
                    chat = chat_completion(
                        messages, model=model, temperature=temperature,
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
                    result_dict["stop_reason"] = "crash"
                    step_dict = {
                        "t": t, "url": step_url, "observation": obs_truncated,
                        "thought": "", "raw_prediction": f"[API_ERROR] {e}",
                        "action": "", "parse_error": "api_error",
                        "prompt_tokens": None, "completion_tokens": None,
                        "latency_ms": 0,
                    }
                    result_dict["steps"].append(step_dict)
                    result_dict["total_steps"] = t
                    break

            if chat is None and result_dict["stop_reason"] == "crash":
                break

            if chat is not None:
                raw_prediction = chat.content
                prompt_tokens = chat.prompt_tokens
                completion_tokens = chat.completion_tokens
                latency_ms = chat.latency_ms

            try:
                action_str = extract_action(template, raw_prediction)
                action_obj = create_id_based_action(action_str)
            except ActionParsingError:
                parse_error = "parse_error"
                action_str = ""
                action_obj = create_none_action()

            action_obj["raw_prediction"] = raw_prediction
            upstream_traj.append(action_obj)

            answer_phrase = template.meta_data["answer_phrase"]
            idx = raw_prediction.find(answer_phrase)
            thought = raw_prediction[:idx].strip() if idx >= 0 else raw_prediction.strip()

            step_dict = {
                "t": t, "url": step_url, "observation": obs_truncated,
                "thought": thought, "raw_prediction": raw_prediction,
                "action": action_str, "parse_error": parse_error,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "latency_ms": latency_ms,
                "is_intervention": t_offset == 0,
            }
            result_dict["steps"].append(step_dict)
            steps_collected.append(step_dict)
            result_dict["total_steps"] = t

            if parse_error is not None:
                recent_parse_failures += 1
                if recent_parse_failures >= _EARLY_STOP_K:
                    result_dict["stop_reason"] = "parse_failures"
                    break
                previous_action_str = "None"
                continue
            recent_parse_failures = 0

            recent_actions.append(action_str)
            if (
                len(recent_actions) >= _EARLY_STOP_K
                and all(
                    action_str != "" and action_str == a
                    for a in recent_actions[-_EARLY_STOP_K:]
                )
            ):
                result_dict["stop_reason"] = "repeating_action"
                break

            if action_obj["action_type"] == ActionTypes.STOP:
                result_dict["stop_reason"] = "agent_stop"
                break

            try:
                obs, _reward, terminated, _truncated, info = env.step(action_obj)
            except Exception:
                result_dict["stop_reason"] = "crash"
                break

            upstream_traj.append({"observation": obs, "info": info})
            previous_action_str = action_str

            if terminated:
                result_dict["stop_reason"] = "env_terminated"
                break
        else:
            result_dict["stop_reason"] = "max_steps"

        # Per SPEC: success/eval_score must be null on crash.
        if result_dict.get("stop_reason") != "crash":
            if upstream_traj and (
                not isinstance(upstream_traj[-1], dict)
                or "action_type" not in upstream_traj[-1]
            ):
                upstream_traj.append(create_stop_action(""))

            try:
                scorer = evaluator_router(config_file)
                score = scorer(upstream_traj, config_file, env.page, env.get_page_client(env.page))
                result_dict["eval_score"] = float(score)
                result_dict["success"] = score == 1.0
            except Exception:
                result_dict["eval_score"] = None
                result_dict["success"] = None

    except Exception as e:
        result_dict["stop_reason"] = "crash"
        result_dict["success"] = None
        result_dict["eval_score"] = None
        result_dict["steps"].append({"crash": str(e), "traceback": traceback.format_exc()})
    finally:
        try:
            env.close()
        except Exception:
            pass

    result_dict["ended_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    label = "env_only" if env_only else "oracle_intervention"
    print(f"  => {label} t*={t_star} stop={result_dict['stop_reason']} | steps={result_dict['total_steps']} | success={result_dict['success']}")

    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result_dict, f, indent=2)

    return result_dict


# ---------------------------------------------------------------------------
# evaluate_batch
# ---------------------------------------------------------------------------

def evaluate_batch(
    condition: str,
    config_files: list[str],
    *,
    out_dir: str,
    **kwargs,
) -> list[dict]:
    """Run run_episode on all config files for a given condition.

    Each result is written to out_dir/<task_id>.json. Returns the list
    of result dicts. Skips tasks that already have output files.
    """
    results = []
    total = len(config_files)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for i, cf in enumerate(config_files):
        with open(cf) as f:
            task_id = json.load(f)["task_id"]

        out_file = out_dir_path / f"{task_id}.json"
        if out_file.exists():
            print(f"[{i+1}/{total}] Skipping task {task_id} (already exists)")
            with open(out_file) as f:
                results.append(json.load(f))
            continue

        print(f"[{i+1}/{total}] Running task {task_id} under condition={condition}")
        try:
            result = run_episode(
                condition, cf, out_path=str(out_file), **kwargs
            )
            results.append(result)
        except Exception:
            print(f"  CRASH on task {task_id}")
            traceback.print_exc()
            results.append({
                "task_id": task_id,
                "condition": condition,
                "stop_reason": "crash",
                "success": None,
                "eval_score": None,
                "total_steps": 0,
                "tokens_used": 0,
            })

    return results


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from a list of run_episode outputs."""
    if not results:
        return {
            "success_rate": 0.0,
            "avg_tokens": 0.0,
            "avg_eval_score": 0.0,
            "by_length_bin": {},
        }

    scored = [r for r in results if r.get("success") is not None]
    success_rate = round(sum(1 for r in scored if r["success"]) / len(scored), 3) if scored else 0.0

    def _total_tokens(r: dict) -> int:
        return sum(
            (s.get("prompt_tokens") or 0) + (s.get("completion_tokens") or 0)
            for s in r.get("steps", [])
        )
    token_vals = [_total_tokens(r) for r in results]
    avg_tokens = round(sum(token_vals) / len(results), 1) if results else 0.0

    eval_scored = [r["eval_score"] for r in results if r.get("eval_score") is not None]
    avg_eval_score = round(sum(eval_scored) / len(eval_scored), 3) if eval_scored else 0.0

    bins: dict[str, list[dict]] = {}
    for r in results:
        total_steps = r.get("total_steps")
        if total_steps is None:
            continue
        bin_key = _length_bin(total_steps)
        if bin_key is None:
            # Trajectory length outside proposal bins (>80 steps) — not aggregated.
            continue
        bins.setdefault(bin_key, []).append(r)

    by_length_bin: dict[str, dict] = {}
    for bin_key in LENGTH_BINS:
        bin_results = bins.get(bin_key, [])
        bin_scored = [r for r in bin_results if r.get("success") is not None]
        bin_success_rate = (
            round(sum(1 for r in bin_scored if r["success"]) / len(bin_scored), 3)
            if bin_scored else 0.0
        )
        bin_eval = [r["eval_score"] for r in bin_results if r.get("eval_score") is not None]
        bin_avg_eval = round(sum(bin_eval) / len(bin_eval), 3) if bin_eval else 0.0

        from collections import Counter
        stop_reasons = Counter(r.get("stop_reason", "unknown") for r in bin_results)

        by_length_bin[bin_key] = {
            "count": len(bin_results),
            "success_rate": bin_success_rate,
            "avg_eval_score": bin_avg_eval,
            "stop_reasons": dict(stop_reasons),
        }

    total_stop = Counter(r.get("stop_reason", "unknown") for r in results)

    return {
        "total": len(results),
        "success_rate": success_rate,
        "avg_tokens": avg_tokens,
        "avg_eval_score": avg_eval_score,
        "stop_reasons": dict(total_stop),
        "by_length_bin": by_length_bin,
    }
