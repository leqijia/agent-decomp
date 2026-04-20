# Evaluation harness interface

The harness is the single entry point for every experimental condition in
the paper. It runs an agent against a WebArena task under a chosen
condition, emits a `trajectories/SPEC.md`-conformant JSON, and scores it
with the upstream evaluator.

This file is the contract the rest of the project codes against: the ReAct
and StateAct agents, every baseline, the oracle pipeline, and the
Experiment 1 replay-and-intervene pipeline all call into the functions
defined here. Changing a signature is a cross-module change — update each
consumer in the same diff.

## What the harness must support

The research design needs two operations, and nothing else belongs in the
harness:

1. Fresh rollout under a condition. A task is run end-to-end with a chosen
   agent variant and a chosen context policy, producing a scored trajectory.
   This covers Experiment 2 (length scaling, binning by trajectory length
   under raw vs oracle), Experiment 3 (raw vs self-generated vs oracle
   external), and every baseline (sliding window, observation masking,
   ACON, AgentDiet, perfect retrieval).

2. Replay and intervene. Given a stored trajectory and a critical step
   `t_star`, the harness re-executes the stored actions against a fresh env
   up to `t_star`, then resumes the agent from that step with its context
   replaced by an externally supplied oracle state. This is Experiment 1
   and is the only operation the other mode cannot cover, because it
   requires the agent's context at one specific step to be surgically
   replaced without re-rolling earlier steps.

Metrics aggregation and batching are thin wrappers over these two. The
harness does not generate oracle states, does not host the agent loop
itself, and does not define the trajectory JSON shape — those live in
`oracle/`, `agent/`, and `trajectories/SPEC.md` respectively.

## Conditions decompose into (agent_variant, context_policy)

A condition name in the existing `VALID_CONDITIONS` set conflates two
independent axes. They should be pulled apart so that adding a new baseline
is a new context policy, not a new branch in a switch statement, and so
that Experiment 3's orthogonal comparison (same context policy across
different agent variants) is expressible.

The axes:

- `agent_variant`: which agent loop runs. Current values: `react`,
  `stateact`. Matches the `agent_variant` field in `trajectories/SPEC.md`.
- `context_policy`: a pure function that decides what text about prior
  steps the agent sees at step `t`.

The mapping from the existing condition names, for a registry in
`harness/conditions.py`:

- `raw`                 -> (`react`,    `identity`)
- `sliding_window`      -> (`react`,    `sliding_window(window_size)`)
- `observation_masking` -> (`react`,    `observation_masking`)
- `acon`                -> (`react`,    `acon`)
- `agentdiet`           -> (`react`,    `agentdiet`)
- `perfect_retrieval`   -> (`react`,    `perfect_retrieval`)
- `env_only`            -> (`react`,    `env_only`)
- `self_generated`      -> (`stateact`, `identity`)
- `oracle_external`     -> (`react`,    `oracle_external(k)`)

`env_only` keeps only the current environment observation plus the
persistent task goal, which is the minimal-context control referenced in
the proposal's decomposition. `oracle_external(k)` regenerates the oracle
state every `k` steps and splices it into the context in place of the raw
step log; `k=1` is the every-step oracle condition, larger `k` amortizes
oracle cost for the ablation in Experiment 3.

## Context policy interface

A context policy is a pure function over the step history. It returns the
string that replaces the default step-log serialization in the agent's
prompt.

```python
from typing import Protocol

class ContextPolicy(Protocol):
    name: str  # short tag, copied into the emitted trajectory's condition metadata

    def __call__(
        self,
        steps_so_far: list[dict],   # prior steps in trajectories/SPEC.md shape
        current_observation: str,   # already truncated to max_obs_length
        intent: str,
    ) -> str: ...
```

Constraints, each tied to a concrete failure mode:

- The policy must be pure. No network calls, no env access, no hidden
  global state. Out-of-band inputs (cached oracle states, retrieval
  corpora) are injected via a closure at construction time. Hidden IO
  breaks reproducibility and makes Experiment 2's bin comparisons
  non-deterministic.
- The policy must be deterministic given its inputs. A policy that
  re-tokenizes or re-orders content on each call invalidates the
  token-budget comparison in Experiment 2.
- The policy must not modify `current_observation`. The env has already
  truncated it to `max_obs_length`; further rewriting belongs on prior
  steps, not on what the agent is looking at right now. Exp 1 and Exp 3
  both compare "same current observation, different context" — a policy
  that mutates the current observation makes that comparison incoherent.

The existing functions in `baselines/trajectory.py`
(`sliding_window_truncate`, `mask_observations`, `serialize_trajectory`)
are already in the right shape to be wrapped as policies.

## run_episode

```python
def run_episode(
    condition: str,                  # key into the condition registry
    config_file: str,                # WebArena task config path
    *,
    model: str = os.environ["AGENT_MODEL"],
    max_steps: int = 75,
    max_obs_length: int = 4096,
    temperature: float = 0.0,
    window_size: int | None = None,          # required iff policy == sliding_window
    oracle_regen_every_k: int | None = None, # required iff policy == oracle_external
    out_path: str | None = None,     # if set, writes SPEC.md-conformant JSON
) -> dict: ...
```

Returns the trajectory dict (identical shape to the on-disk JSON). Raises
`ValueError` on unknown condition or on a missing required policy kwarg.

Internally this is the upstream WebArena loop from
`references/webarena/run.py` with one hook: after each `env.step`, the
harness calls `context_policy(steps_so_far, obs, intent)` and passes the
result into the agent's prompt assembler in place of the default step-log
serialization. The upstream `early_stop` and `evaluator_router` functions
are used as-is for the stop conditions and for scoring.

The returned trajectory must satisfy `trajectories/SPEC.md`, including the
`stop_reason`, `success`, and `eval_score` fields populated from
`evaluator_router(config_file)(trajectory, ...)`.

## run_intervention

```python
def run_intervention(
    trajectory_path: str,            # stored SPEC.md-conformant JSON
    t_star: int,                     # 1-indexed step at which to intervene
    replacement_context: str,        # oracle state, already serialized
    *,
    model: str = os.environ["AGENT_MODEL"],
    out_path: str | None = None,
) -> dict: ...
```

Semantics:

1. Load the stored trajectory. Open a fresh WebArena env from its
   `config_file`.
2. Replay steps `1..t_star - 1` by feeding each stored `action` string
   through `browser_env.actions.create_id_based_action` and then
   `env.step`. If a replay step raises `ActionParsingError`, or if the
   live `page.url` after a step diverges from the stored `url` for that
   step, abort the run with `stop_reason = "replay_desync"` and return the
   partial trajectory. Desync is load-bearing: if replay silently drifts,
   the intervention measures the wrong state.
3. At step `t_star`, run the agent with `replacement_context` substituted
   for the usual step-log serialization. The current observation is
   whatever the replayed env produces at that point; do not attempt to
   restore the stored observation, since element ids may have shifted.
4. Continue the normal loop from `t_star` until stop, max_steps, or env
   termination.
5. Score with `evaluator_router` and emit a SPEC.md-conformant trajectory
   with `agent_variant = "oracle_external"` and an extra top-level field
   `intervention = {"t_star": t_star, "source_trajectory": trajectory_path}`
   so downstream analysis can join the intervened run back to its source.

The harness takes `replacement_context` as an already-serialized string and
does not call the oracle model itself. This keeps the harness free of
OpenRouter calls for Experiment 1 and lets oracle generation be cached,
audited, and re-run independently of the replay.

A new `stop_reason` value `replay_desync` is introduced by this operation
and must be added to the allowed set in `trajectories/SPEC.md` alongside
the six existing values.

## Batch evaluation and metrics

```python
def evaluate_batch(
    condition: str,
    config_files: list[str],
    *,
    out_dir: str,
    **kwargs,  # forwarded to run_episode
) -> list[dict]: ...

def compute_metrics(results: list[dict]) -> dict: ...
```

`compute_metrics` already exists in `harness/evaluator.py` and returns
`{"success_rate": float, "avg_tokens": float}`. Two changes are needed for
it to serve the three experiments:

- Include `eval_score` in each result entry, not only the boolean
  `success`. Experiment 2's context gap `Gamma(L) = oracle_acc - raw_acc`
  per length bin uses the continuous score when the evaluator provides one.
- Include `length_bin` (derived from `total_steps`) in each entry so Exp 2
  can group without a second pass.

Batch runs are embarrassingly parallel across tasks. Parallelism should
live in an external driver (multiple processes), not inside the harness.
Sharing a Playwright instance across threads is a known source of flake
and is not worth the complexity for an offline evaluation job.

## Out of scope for this module

The harness does not generate oracle states; it consumes serialized oracle
text produced by `oracle/generate_oracle.py`. The harness does not define
the agent loop internals; it imports from `agent/react_agent.py` and
(eventually) `agent/stateact_agent.py`. The harness does not define the
trajectory JSON shape; it writes records conforming to
`trajectories/SPEC.md`. The harness does not manage container lifecycles;
WebArena provides no snapshot primitive, and every episode opens a fresh
browser context against the same running services.

## Suggested implementation order

1. Add the condition registry and the `ContextPolicy` protocol. Wrap the
   two existing functions in `baselines/trajectory.py` as policies.
2. Implement `run_episode` for `raw`, `sliding_window`, and
   `observation_masking` first — these only need the agent loop plus a
   policy swap and have no oracle dependency.
3. Add `oracle_external` and `self_generated` once the ReAct and StateAct
   agents are in place.
4. Implement `run_intervention` last. It depends on a working
   `run_episode`, on stored trajectories, and on oracle states being
   produced by the oracle pipeline.
