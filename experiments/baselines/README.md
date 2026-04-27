# Baselines

Per-method runs of the five context-management baselines from the MMLA proposal §3.4, plus the two reference Exp3 conditions (`raw`, `self_generated`) for direct comparison. Every method runs against the same Exp3 task suite (`experiments/exp3/raw/manifest.csv`) through `harness.evaluator.run_episode`, so the resulting metrics are directly comparable.

## Layout

```
experiments/baselines/
├── README.md                       (this file)
├── baselines_table.md              (deliverable; produced by build_baselines_table.py)
├── baselines_table.json            (structured form of the same)
├── sliding_window/
│   ├── data/                       (per-task trajectory JSON)
│   ├── manifest.csv                (build_condition_manifest.py)
│   ├── results.json                (build_condition_manifest.py)
│   ├── skipped.csv                 (build_condition_manifest.py)
│   └── metrics.json                (compute_condition_metrics.py)
├── observation_masking/            (same five files)
├── acon/                           (same five files)
├── agentdiet/                      (same five files)
└── perfect_retrieval/              (same five files)
```

The reference Exp3 conditions live under `experiments/exp3/<condition>/` with the same five files; `build_baselines_table.py` pulls metrics from both trees.

## What each method does (one-liner)

| Method                | Mechanism                                                                                                                | Implementation                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------- |
| `sliding_window`      | Truncate trajectory to the last *N* tokens before each decision.                                                         | `baselines/trajectory.sliding_window_truncate` |
| `observation_masking` | Replace older observations with `[MASKED]`; keep the latest observation and all reasoning/action tokens.                  | `baselines/trajectory.mask_observations`    |
| `acon`                | LLM-based trajectory compression (ACON paper). Uses `OPENROUTER_API_KEY` + `ACON_MODEL`; falls back to a stub offline.    | `baselines/acon/`                           |
| `agentdiet`           | Heuristic step filter (drops "useless" steps by token-budget rule).                                                       | `baselines/agentdiet.agentdiet_filter`      |
| `perfect_retrieval`   | Top-*k* dense retrieval over past steps. **Live-query semantics — see deviation below.**                                  | `baselines/perfect_retrieval.py` + `harness.evaluator._perfect_retrieval_policy` |

## Perfect-retrieval live-query deviation

`baselines/perfect_retrieval.py` was originally written for **post-hoc replay**: at step *t* it queries with the *future logged action a_t* (an oracle quantity) against the past-step corpus. That formulation is an upper bound on retrieval quality but cannot run live, because the future action is unknown at decision time.

For the harness path, `harness.evaluator._perfect_retrieval_policy` keeps the same retrieval mechanism (cosine over chunked past steps via `deterministic_embed_fn`) but replaces the query with **`intent + current_observation`** — a quantity that *is* available live. This is documented in the policy's docstring and surfaces in the baselines-table notes.

If reviewers ask for the strict-oracle version, it fits naturally into `harness.evaluator.run_intervention` (the Exp1 path), which has access to the logged trajectory.

The current embedding function (`baselines.perfect_retrieval.deterministic_embed_fn`) is a SHA256-seeded random projection — reproducible and dependency-light, but not semantic. Swap to `text-embedding-3-small` (OpenAI) for paper-grade results; the contract is unchanged (`Callable[[list[str]], np.ndarray]`).

## How to (re)generate a single method

```bash
# 1) run rollouts
python scripts/run_condition_batch.py --condition <method> \
    --task-list experiments/exp3/raw/manifest.csv \
    --out-dir   experiments/baselines/<method>/data \
    --model     qwen/qwen3.5-27b

# 2) build the manifest sidecar
python scripts/build_condition_manifest.py \
    --data-dir experiments/baselines/<method>/data \
    --out-dir  experiments/baselines/<method> \
    --condition <method> --experiment-id baselines

# 3) compute aggregate metrics
python scripts/compute_condition_metrics.py \
    --results experiments/baselines/<method>/results.json \
    --out     experiments/baselines/<method>/metrics.json
```

For `perfect_retrieval`, optionally tune `k` via `--perfect-retrieval-k 3` (default).

## How to regenerate the deliverable table

```bash
python scripts/build_baselines_table.py
```

Reads `metrics.json` from each `<method>/` plus the two Exp3 reference conditions and emits `baselines_table.md` and `baselines_table.json`. Methods missing a `metrics.json` are skipped with a warning, so the table can be regenerated at any point during the experiment.
