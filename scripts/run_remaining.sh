#!/usr/bin/env bash
# Run the remaining experiments end-to-end.
#
# Run on the VM after `git pull`:
#   bash scripts/run_remaining.sh [--skip-gpt52]
#
# Stages, in order:
#   STAGE 1  Generate t*-specific oracle states for all annotated tasks
#            (cheap; ~$10 total for ~180 calls @ Sonnet 4.6)
#   STAGE 2  Run Exp 1 oracle/env-only intervention at every annotated t*
#            (Qwen agent; ~$30 total for ~360 runs)
#   STAGE 3  Recompute kappa + alpha (primary, upper bound, pooled)
#   STAGE 4  Run all 5 baselines on the Exp 3 raw manifest (Qwen; ~$50)
#   STAGE 5  Run Exp 3 self-generated condition (StateAct; ~$50)
#   STAGE 6  Run Exp 3 oracle-external condition with prompt caching (~$80)
#   STAGE 7  Run GPT-5.2 generalizability on 100-task subset (Exp 1 only;
#            ~$300, capped at $400). Skip with --skip-gpt52 if budget is tight.
#   STAGE 8  Compute Gamma(L), delta_synth, baselines table.
#
# Total estimated spend: ~$500 (Qwen ~$130, oracle ~$90, GPT-5.2 ~$300).
# Idempotent: every stage skips work whose output already exists.

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_GPT52=0
for a in "$@"; do
  case "$a" in
    --skip-gpt52) SKIP_GPT52=1 ;;
  esac
done

# Sites must be up before any agent run; bring up the containers if not.
docker start shopping shopping_admin gitlab forum >/dev/null 2>&1 || true

RAW_DATA_DIR=trajectories/data
ORACLE_EXTERNAL_DATA_DIR=trajectories/oracle_external
LOG=experiments/run_remaining.log
mkdir -p experiments
echo "==== run_remaining started $(date -u +%FT%TZ) ====" | tee -a "$LOG"

stage() { echo -e "\n=== STAGE $1: $2 ===" | tee -a "$LOG"; }

# -----------------------------------------------------------------------------
stage "PRE" "Preflight dependencies, WebArena, and OpenRouter"
PREFLIGHT_ARGS=()
if [ "$SKIP_GPT52" -eq 1 ]; then
  PREFLIGHT_ARGS+=(--skip-gpt52)
fi
python scripts/preflight_remaining.py "${PREFLIGHT_ARGS[@]}" 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
stage 0 "Refresh Exp 3 raw manifest from current raw trajectories"
python scripts/build_exp3_raw_manifest.py \
    --data-dir "$RAW_DATA_DIR" \
    --out-dir experiments/exp3/raw \
    --condition raw \
    --experiment-id exp3 2>&1 | tee -a "$LOG"
python scripts/compute_condition_metrics.py \
    --results experiments/exp3/raw/results.json \
    --out     experiments/exp3/raw/metrics.json 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
stage 1 "Generate t*-specific oracle states for all 4 annotators"
python scripts/generate_tstar_oracles.py 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
stage 2 "Run Exp 1 oracle + env_only at every annotated t*"
python scripts/run_oracle_baseline.py 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
stage 3 "Recompute kappa, primary alpha, upper-bound alpha"
python scripts/build_tstar_resolved.py 2>&1 | tee -a "$LOG"
python scripts/compute_kappa.py        2>&1 | tee -a "$LOG"
python scripts/compute_alpha.py        2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
stage 4 "Run 5 baselines on Exp 3 raw manifest"
for COND in sliding_window observation_masking acon agentdiet perfect_retrieval; do
  echo "--- baseline: $COND ---" | tee -a "$LOG"
  python scripts/run_condition_batch.py --condition "$COND" \
      --task-list experiments/exp3/raw/manifest.csv \
      --out-dir   "experiments/baselines/${COND}/data" \
      --model     qwen/qwen3.5-27b \
      --rerun-crashes 2>&1 | tee -a "$LOG"
  python scripts/build_condition_manifest.py \
      --data-dir "experiments/baselines/${COND}/data" \
      --out-dir  "experiments/baselines/${COND}" \
      --condition "$COND" --experiment-id baselines 2>&1 | tee -a "$LOG"
  python scripts/compute_condition_metrics.py \
      --results "experiments/baselines/${COND}/results.json" \
      --out     "experiments/baselines/${COND}/metrics.json" 2>&1 | tee -a "$LOG"
done

# -----------------------------------------------------------------------------
stage 5 "Run Exp 3 self-generated state (StateAct) on full task suite"
python scripts/run_condition_batch.py --condition self_generated \
    --task-list experiments/exp3/raw/manifest.csv \
    --out-dir   experiments/exp3/self_generated/data \
    --model     qwen/qwen3.5-27b \
    --rerun-crashes 2>&1 | tee -a "$LOG"
python scripts/build_condition_manifest.py \
    --data-dir experiments/exp3/self_generated/data \
    --out-dir  experiments/exp3/self_generated \
    --condition self_generated --experiment-id exp3 2>&1 | tee -a "$LOG"
python scripts/compute_condition_metrics.py \
    --results experiments/exp3/self_generated/results.json \
    --out     experiments/exp3/self_generated/metrics.json 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
stage 6 "Run Exp 3 oracle-external (prompt-cached oracle) on full suite"
python scripts/run_condition_batch.py --condition oracle_external \
    --task-list experiments/exp3/raw/manifest.csv \
    --out-dir   "$ORACLE_EXTERNAL_DATA_DIR" \
    --model     qwen/qwen3.5-27b \
    --rerun-crashes 2>&1 | tee -a "$LOG"
python scripts/build_condition_manifest.py \
    --data-dir "$ORACLE_EXTERNAL_DATA_DIR" \
    --out-dir  experiments/exp3/oracle_external \
    --condition oracle_external --experiment-id exp3 2>&1 | tee -a "$LOG"
python scripts/compute_condition_metrics.py \
    --results experiments/exp3/oracle_external/results.json \
    --out     experiments/exp3/oracle_external/metrics.json 2>&1 | tee -a "$LOG"

# -----------------------------------------------------------------------------
if [ "$SKIP_GPT52" -eq 0 ]; then
  stage 7 "Run Exp 1 generalizability on 100-task subset with GPT-5.2"
  python scripts/run_generalizability.py \
      --subset-size 100 \
      --model openai/gpt-5.2 \
      --max-cost-usd 400 2>&1 | tee -a "$LOG"
else
  echo "Skipping STAGE 7 (--skip-gpt52)" | tee -a "$LOG"
fi

# -----------------------------------------------------------------------------
stage 8 "Final analysis: Gamma(L), delta_synth, baselines table"
python scripts/compute_gamma.py        2>&1 | tee -a "$LOG"
python scripts/build_baselines_table.py 2>&1 | tee -a "$LOG"
# delta_synth is computed from exp3 oracle_external + self_generated metrics
python -c "
import json, sys
sys.path.insert(0,'.')
from harness.metrics import compute_delta_synth
oracle = json.load(open('experiments/exp3/oracle_external/results.json'))['results']
self_g = json.load(open('experiments/exp3/self_generated/results.json'))['results']
out = compute_delta_synth(oracle, self_g)
json.dump(out, open('experiments/delta_synth_results.json','w'), indent=2)
print('Wrote experiments/delta_synth_results.json')
print(out)
" 2>&1 | tee -a "$LOG"

echo -e "\n==== run_remaining finished $(date -u +%FT%TZ) ====" | tee -a "$LOG"
echo "Final cost log: $(jq '.total_cost_usd' oracle/cost_log.json 2>/dev/null) (oracle) + see budget.json (agent)"
