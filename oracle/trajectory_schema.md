# Trajectory JSON Schema

**Status: CONFIRMED — sourced from trajectories/SPEC.md**

---

## Top-level fields

| Field              | Type         | Description                                              |
|--------------------|--------------|----------------------------------------------------------|
| task_id            | string       | Unique identifier for the task                           |
| config_file        | string       | Path to the WebArena task config used                    |
| sites              | list[string] | WebArena sites involved (e.g. ["shopping", "reddit"])    |
| intent             | string       | Natural language task description                        |
| start_url          | string       | URL the agent starts from                                |
| model              | string       | Model used (e.g. "qwen/Qwen3.5-27B")                    |
| agent_variant      | string       | Agent class/config (e.g. "standard", "acon")             |
| observation_type   | string       | How observations are captured (e.g. "accessibility_tree")|
| action_set_tag     | string       | Action set used (e.g. "id_accessibility_tree")           |
| max_steps          | int          | Step budget for the episode                              |
| temperature        | float        | Sampling temperature                                     |
| max_obs_length     | int          | Max characters in each observation                       |
| started_at         | string       | ISO 8601 timestamp when episode began                    |
| ended_at           | string       | ISO 8601 timestamp when episode ended                    |
| total_steps        | int          | Number of steps actually taken                           |
| stop_reason        | string       | Why the episode stopped ("max_steps", "done", "error")   |
| success            | bool         | Whether the task was completed successfully              |
| eval_score         | float        | Continuous evaluation score (0.0–1.0)                    |
| steps              | list[dict]   | Ordered list of step objects (see below)                 |

---

## Step object fields

| Field             | Type   | Description                                                  |
|-------------------|--------|--------------------------------------------------------------|
| t                 | int    | Step number (1-indexed)                                      |
| url               | string | URL of the page at this step                                 |
| observation       | string | Agent's observation (accessibility tree or screenshot text)  |
| thought           | string | Agent's reasoning before acting                              |
| raw_prediction    | string | Raw model output before action parsing                       |
| action            | string | Parsed action executed                                       |
| parse_error       | string | Error message if action parsing failed, else null            |
| prompt_tokens     | int    | Tokens in the prompt for this step                           |
| completion_tokens | int    | Tokens in the model completion for this step                 |
| latency_ms        | int    | Time taken for the model call in milliseconds                |

---

## DOM snapshots

**DOM snapshots are NOT stored in trajectory files.**

The oracle fetches them live from the WebArena environment via CDP
(Chrome DevTools Protocol) at oracle generation time. This means:

- Oracle generation requires the WebArena Docker environment to be running
- The DOM captured reflects the ground-truth page state at step t,
  which may differ from the agent's `observation` field (stale or truncated)
- The CDP fetch is handled separately from this pipeline — see Rocky for details

---

## Compatibility with generate_oracle.py

`build_prompt()` in `oracle/generate_oracle.py` reads these step fields:

    t, thought, action, observation

All four are present in the confirmed schema with the same names.
**No field name changes needed** when switching from dummy to real trajectories.

The `task_goal` argument to `build_prompt()` maps to `intent` at the top level.

---

## Implementation notes

**Field name alignment** — All step fields read by `build_prompt()` (`t`, `thought`,
`action`, `observation`) match the confirmed schema exactly. No remapping needed.

**Evaluator error objects** — The harness appends a dict with key `"evaluator_error"`
(not `"t"`) as the last element of `steps[]` when scoring fails. All pipeline code
that iterates over steps must filter these out:

    steps = [s for s in traj["steps"] if "t" in s]

**DOM snapshots not stored** — `observation` is the agent's accessibility tree at the
time of the step, not a ground-truth DOM snapshot. For oracle generation the interim
fallback is `use_observation=True` in `get_live_dom()`, which passes the stored
observation as the DOM snapshot until CDP access is confirmed with Rocky.

**total_steps accuracy** — `total_steps` counts actual agent steps, excluding the
evaluator error object. The step numbers `t` run 1 … total_steps inclusive.

**Step counts per task (current batch)**

| Task | total_steps | Steps with oracle states |
|------|-------------|--------------------------|
| 27   | 30          | 1, 8, 16, 23, 30         |
| 62   | 18+         | 1, 5, 10, 14, 18         |
| 114  | 19+         | 1, 6, 10, 14, 19         |
| 130  | 27+         | 1, 8, 14, 20, 27         |
| 199  | 30          | 1, 8, 16, 23, 30         |
| 322  | 30          | 1, 8, 16, 23, 30         |
| 332  | 24+         | 1, 7, 12, 18, 24         |
| 401  | 26+         | 1, 7, 14, 20, 26         |

Oracle states are k=5 evenly spaced steps selected by `evenly_spaced_steps()` in
`oracle/batch_generate.py`, always including step 1 and the final step.
