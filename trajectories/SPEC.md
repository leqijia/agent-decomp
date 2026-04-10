# Trajectory JSON schema

On-disk format for one agent episode. Emitted by `agent/react_agent.py` and
read by the oracle pipeline, the harness, every baseline, annotators, and the
Experiment 1 replay-and-intervene pipeline.

If you need a field that isn't here, ping Rocky — don't just add one, other
people's code depends on the shape.

## Example

```json
{
  "task_id": 156,
  "config_file": "config_files/156.json",
  "sites": ["gitlab"],
  "intent": "Create an issue in the a11y-webring.club repo",
  "start_url": "http://172.185.52.29:8023",

  "model": "qwen/qwen3-32b",
  "agent_variant": "react",
  "observation_type": "accessibility_tree",
  "action_set_tag": "id_accessibility_tree",
  "max_steps": 30,
  "temperature": 0.0,
  "max_obs_length": 1920,

  "started_at": "2026-04-09T21:15:03Z",
  "ended_at":   "2026-04-09T21:17:42Z",
  "total_steps": 2,
  "stop_reason": "agent_stop",
  "success": true,
  "eval_score": 1.0,

  "steps": [
    {
      "t": 1,
      "url": "http://172.185.52.29:8023/",
      "observation": "[4] RootWebArea 'Projects · Dashboard' focused: True\n  [287] link 'a11y-webring.club'",
      "thought": "The dashboard lists projects. I open the a11y-webring.club repo.",
      "raw_prediction": "Let's think... In summary, the next action I will perform is ```click [287]```",
      "action": "click [287]",
      "parse_error": null,
      "prompt_tokens": 2104,
      "completion_tokens": 57,
      "latency_ms": 1842
    },
    {
      "t": 2,
      "url": "http://172.185.52.29:8023/root/a11y-webring.club",
      "observation": "...",
      "thought": "I have the project page open. Task done.",
      "raw_prediction": "```stop [done]```",
      "action": "stop [done]",
      "parse_error": null,
      "prompt_tokens": 2211,
      "completion_tokens": 12,
      "latency_ms": 912
    }
  ]
}
```

## Top-level fields

`task_id`, `config_file`, `sites`, `intent`, `start_url` — mirror the upstream
WebArena config file (`references/webarena/config_files/test.raw.json`) so
trajectory ↔ config is a trivial join. Exp 1 replay re-opens `config_file`
with `ScriptBrowserEnv.reset(options={"config_file": ...})`.

`model` — OpenRouter slug (`qwen/qwen3-32b`, `openai/gpt-5.2`). Needed to slice
the Experiment 1 generalizability comparison.

`agent_variant` — `react` | `stateact` | `oracle_external` | `teacher_forcing`.
Needed to slice Experiment 3.

`observation_type` — `accessibility_tree` | `html` | `image`. Constant per
episode (so it lives here, not per step).

`action_set_tag` — `id_accessibility_tree` | `playwright`. Constant per episode.

`max_steps`, `temperature`, `max_obs_length` — run config. `max_obs_length`
matches upstream's `--max_obs_length` (default 1920 tokens), applied to each
observation before prompting.

`started_at`, `ended_at` — ISO8601 UTC.

`total_steps` — must equal `len(steps)`.

`stop_reason` — one of:
- `agent_stop` — agent emitted a `stop [...]` action
- `max_steps` — hit the step cap
- `parse_failures` — K consecutive parse failures (upstream default K=3, see
  `references/webarena/run.py:161` `early_stop`)
- `repeating_action` — K consecutive identical actions (same source)
- `env_terminated` — `env.step` returned `terminated=True`
- `crash` — unhandled exception; `success` is `null` in this case

`success`, `eval_score` — from `evaluator_router(config_file)(...)`. `null` iff
`stop_reason == "crash"` or the evaluator couldn't run.

## Step fields

`t` — 1-indexed, must satisfy `steps[i].t == i + 1`.

`url` — page URL before the action ran.

`observation` — the text the agent was prompted with at this step, after any
`max_obs_length` truncation. We store the truncated form so the oracle and
replay reproduce the agent's exact view.

`thought` — reasoning text extracted from the model output (empty string if
none was produced).

`raw_prediction` — full model output before parsing. Always stored, even on
parse failure, so we can audit Qwen3 `<think>` blocks and diagnose parse
errors without re-running.

`action` — canonical action string. Empty string iff `parse_error != null`.
Grammar is upstream's `id_accessibility_tree` action set — see
`references/webarena/agent/prompts/raw/p_cot_id_actree_2s.py` for the full
list. The short version:

```
click [id]
type [id] [content] [press_enter_after]
hover [id]
press [key_comb]
scroll [up|down]
goto [url]
go_back
go_forward
new_tab
tab_focus [index]
close_tab
stop [answer]
```

Every non-empty `action` must round-trip through
`browser_env.actions.create_id_based_action(action)` — that's how Exp 1 replay
will re-execute stored actions against a fresh env. If you invent a new action
format, replay breaks.

`parse_error` — error message if action parsing failed; `null` otherwise.

`prompt_tokens`, `completion_tokens`, `latency_ms` — from the OpenRouter
`usage` field + wall clock. Needed for the "tokens per condition" efficiency
column in Table 1 of the proposal. Nullable if the provider didn't return usage.

## What's deliberately not stored

**No per-step ground-truth DOM.** The oracle fetches it live from the WebArena
env via CDP (`Accessibility.getFullAXTree`) at oracle generation time — either
from the live page in Exp 3 or after replaying to step t* in Exp 1. Storing it
per step would bloat files and go stale. If we ever need persisted ground-truth
DOM for offline analysis, put it in a sidecar file, not here.

**No parsed action sub-fields** (`element_id`, `text`, `direction`, `answer`,
…). All re-derivable from the canonical `action` string via
`create_id_based_action`. Duplicated state goes stale.

**No `storage_state`, `require_login`, `intent_template`, etc.** Already in
the WebArena config file referenced by `config_file`. Don't copy upstream's
config into every record.

## Validation rules

A file is conformant iff:

1. It parses as JSON.
2. All top-level fields above (except `temperature`, `max_obs_length`,
   `prompt_tokens`/`completion_tokens`/`latency_ms`) are present.
3. `len(steps) == total_steps` and `steps[i].t == i + 1`.
4. `stop_reason` is one of the six values listed.
5. `action == ""` iff `parse_error is not None`.
6. Every non-empty `action` passes `create_id_based_action(action)` without
   raising `ActionParsingError`.

A validator enforcing these will live at `trajectories/validate.py` (TBD).
