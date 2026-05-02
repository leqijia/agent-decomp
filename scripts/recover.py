#!/usr/bin/env python3
"""Recovery orchestrator after the OpenRouter-out batch.

What it does, in order, with a hard total-spend cap (default $50):

  Stage 0  smoke-check: read budget+cost-log, alert if first chunk produces
           no spend (signals OpenRouter still empty).
  Stage 1  quarantine every file with stop_reason='crash' under
           {trajectories/data, experiments/oracle_baseline, experiments/ablations,
            experiments/exp3/oracle_external/data, experiments/sensitivity}/_crashed_<ts>/
  Stage 2  re-roll trajectories (chunked; cost-checked between chunks)
  Stage 3  generate t*-specific oracle states (Sonnet)
  Stage 4  Exp 1 oracle/env-only interventions
  Stage 5  free analysis: build_tstar_resolved, kappa, alpha
  Stage 6  five baselines (sliding_window / obs_masking / acon / agentdiet / perfect_retrieval)
  Stage 7  Exp 3 self-generated state (StateAct full suite)
  Stage 8  Exp 3 oracle-external (refills the 21 quarantined + the rest of the manifest)
  Stage 9  ablations (per-field oracle drops)
  Stage 10 sensitivity (25/50/75% interventions)
  Stage 11 free analysis: gamma, baselines table, delta_synth

Stops cleanly between stages once cumulative spend on this invocation
exceeds --cap. Re-run with the same command to continue (every stage is
idempotent — outputs already on disk are skipped).

Usage:
  python scripts/recover.py                         # default $50 cap
  python scripts/recover.py --cap 100               # higher cap
  python scripts/recover.py --skip-quarantine       # resume mode (don't re-quarantine)
  python scripts/recover.py --chunk-size 20         # bigger trajectory chunks
  python scripts/recover.py --only-stages 2,3,4     # run a subset
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOG_DIR / f"recover_{RUN_ID}.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    with open(LOGFILE, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def _read_total(path, key="total_cost_usd"):
    try:
        return float(json.load(open(path)).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def cost_snapshot():
    """(qwen, oracle, total) cumulative dollars across both ledgers."""
    q = _read_total(REPO / "budget.json")
    o = _read_total(REPO / "oracle/cost_log.json")
    return q, o, q + o


def fmt_cost():
    q, o, t = cost_snapshot()
    return f"qwen=${q:.4f} oracle=${o:.4f} total=${t:.4f}"


def stage_blocked(label, baseline, cap):
    spent = cost_snapshot()[2] - baseline
    if spent >= cap:
        log(f"SKIP [{label}]: cap reached (${spent:.2f} ≥ ${cap:.2f})")
        return True
    return False


def run_subproc(label, argv, baseline, cap, env_extra=None):
    """Run a child process; capture all output to LOGFILE; report cost delta."""
    if stage_blocked(label, baseline, cap):
        return "skipped", 0.0

    t_before = cost_snapshot()[2]
    log(f"--- RUN [{label}] ---")
    log(f"  before: {fmt_cost()}  this-run-spent=${t_before-baseline:.4f}")
    log(f"  argv: {' '.join(str(a) for a in argv)}")
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    with open(LOGFILE, "a") as f:
        f.write(f"\n>>>>> stdout/stderr of [{label}] >>>>>\n")
        f.flush()
        proc = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(REPO), env=env)
        f.write(f"<<<<< end of [{label}] (rc={proc.returncode}) <<<<<\n\n")

    t_after = cost_snapshot()[2]
    delta = t_after - t_before
    log(f"  after:  {fmt_cost()}  this-run-spent=${t_after-baseline:.4f}  "
        f"delta=${delta:.4f}  rc={proc.returncode}")
    return ("ok" if proc.returncode == 0 else f"rc={proc.returncode}"), delta


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------

QUARANTINE_DIRS = {
    "trajectories/data":                          "trajectory",
    "experiments/oracle_baseline":                "intervention",
    "experiments/exp3/oracle_external/data":      "exp3-oracle-external",
    "experiments/sensitivity":                    "sensitivity",
}


def quarantine_crashes(stamp):
    """Move all stop_reason='crash' files into _crashed_<stamp>/ subdirs.
    Returns dict {location: count_moved}.
    """
    summary = {}
    for loc in QUARANTINE_DIRS:
        path = REPO / loc
        if not path.is_dir():
            summary[loc] = 0
            continue
        qdir = path / f"_crashed_{stamp}"
        qdir.mkdir(parents=True, exist_ok=True)
        moved = 0
        for fp in path.glob("*.json"):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            if d.get("stop_reason") == "crash":
                shutil.move(str(fp), str(qdir / fp.name))
                moved += 1
        if moved == 0:
            try: qdir.rmdir()
            except OSError: pass
        summary[loc] = moved

    # Ablations: every existing result file is currently a crash.
    abl = REPO / "experiments/ablations"
    qdir = abl / f"_crashed_{stamp}"
    qdir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for fp in abl.glob("*_result.json"):
        try:
            d = json.load(open(fp))
            if d.get("stop_reason") != "crash":
                continue  # keep any non-crash result if present
        except Exception:
            continue
        shutil.move(str(fp), str(qdir / fp.name))
        moved += 1
    if moved == 0:
        try: qdir.rmdir()
        except OSError: pass
    summary["experiments/ablations"] = moved
    return summary


def crashed_trajectory_ids():
    """Task IDs whose only trajectory in trajectories/data is a quarantined crash.
    Returns sorted list of ints. Excludes any task ID that has a valid
    (non-quarantined, non-crashed) trajectory file present.
    """
    data = REPO / "trajectories/data"
    valid = set()
    for fp in data.glob("*.json"):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if d.get("stop_reason") != "crash":
            tid = d.get("task_id") or fp.stem
            try: valid.add(int(tid))
            except (TypeError, ValueError): pass
    crashed = set()
    for fp in data.rglob("_crashed_*/*.json"):
        try:
            d = json.load(open(fp))
            tid = d.get("task_id") or fp.stem
            crashed.add(int(tid))
        except Exception:
            continue
    return sorted(crashed - valid)


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

def baseline_stage(cond):
    out_dir = f"experiments/baselines/{cond}/data"
    out_root = f"experiments/baselines/{cond}"
    return [
        (f"baseline-{cond}-runs",
         ["python", "scripts/run_condition_batch.py",
          "--condition", cond,
          "--task-list", "experiments/exp3/raw/manifest.csv",
          "--out-dir", out_dir,
          "--model", "qwen/qwen3.5-27b"]),
        (f"baseline-{cond}-manifest",
         ["python", "scripts/build_condition_manifest.py",
          "--data-dir", out_dir,
          "--out-dir", out_root,
          "--condition", cond,
          "--experiment-id", "baselines"]),
        (f"baseline-{cond}-metrics",
         ["python", "scripts/compute_condition_metrics.py",
          "--results", f"{out_root}/results.json",
          "--out", f"{out_root}/metrics.json"]),
    ]


def condition_stage(cond, root):
    out_dir = f"{root}/data"
    return [
        (f"{cond}-runs",
         ["python", "scripts/run_condition_batch.py",
          "--condition", cond,
          "--task-list", "experiments/exp3/raw/manifest.csv",
          "--out-dir", out_dir,
          "--model", "qwen/qwen3.5-27b"]),
        (f"{cond}-manifest",
         ["python", "scripts/build_condition_manifest.py",
          "--data-dir", out_dir,
          "--out-dir", root,
          "--condition", cond,
          "--experiment-id", "exp3"]),
        (f"{cond}-metrics",
         ["python", "scripts/compute_condition_metrics.py",
          "--results", f"{root}/results.json",
          "--out", f"{root}/metrics.json"]),
    ]


# ordered (stage_id, label, argv) tuples; stages 1-2 are handled outside this list
PIPELINE = []

# Stage 3
PIPELINE.append((3, "tstar-oracles", ["python", "scripts/generate_tstar_oracles.py"]))

# Stage 4
PIPELINE.append((4, "interventions", ["python", "scripts/run_oracle_baseline.py"]))

# Stage 5 — free
PIPELINE.append((5, "tstar-resolved", ["python", "scripts/build_tstar_resolved.py"]))
PIPELINE.append((5, "compute-kappa", ["python", "scripts/compute_kappa.py"]))
PIPELINE.append((5, "compute-alpha", ["python", "scripts/compute_alpha.py"]))

# Stage 6 — baselines
for cond in ["sliding_window", "observation_masking", "acon", "agentdiet", "perfect_retrieval"]:
    for label, argv in baseline_stage(cond):
        PIPELINE.append((6, label, argv))

# Stage 7 — self-generated state
for label, argv in condition_stage("self_generated", "experiments/exp3/self_generated"):
    PIPELINE.append((7, label, argv))

# Stage 8 — oracle external
for label, argv in condition_stage("oracle_external", "experiments/exp3/oracle_external"):
    PIPELINE.append((8, label, argv))

# Stage 9 — ablations
PIPELINE.append((9, "ablations", ["python", "scripts/run_ablations.py"]))

# Stage 10 — sensitivity
PIPELINE.append((10, "sensitivity", ["python", "scripts/run_sensitivity.py"]))

# Stage 11 — final analysis
PIPELINE.append((11, "compute-gamma", ["python", "scripts/compute_gamma.py"]))
PIPELINE.append((11, "compute-alpha-human", ["python", "scripts/compute_alpha_human.py"]))
PIPELINE.append((11, "baselines-table", ["python", "scripts/build_baselines_table.py"]))
PIPELINE.append((11, "delta-synth", [
    "python", "-c",
    "import json, sys; sys.path.insert(0,'.'); "
    "from harness.metrics import compute_delta_synth; "
    "oracle=json.load(open('experiments/exp3/oracle_external/results.json'))['results']; "
    "self_g=json.load(open('experiments/exp3/self_generated/results.json'))['results']; "
    "out=compute_delta_synth(oracle, self_g); "
    "json.dump(out, open('experiments/delta_synth_results.json','w'), indent=2); "
    "print('delta_synth:', out)"
]))


# ---------------------------------------------------------------------------
# Final post-pipeline audit
# ---------------------------------------------------------------------------

AUDIT_DIRS = [
    "trajectories/data",
    "experiments/oracle_baseline",
    "experiments/exp3/oracle_external/data",
    "experiments/exp3/self_generated/data",
    "experiments/exp3/raw/data",
    "experiments/baselines/sliding_window/data",
    "experiments/baselines/observation_masking/data",
    "experiments/baselines/acon/data",
    "experiments/baselines/agentdiet/data",
    "experiments/baselines/perfect_retrieval/data",
    "experiments/sensitivity",
    "experiments/ablations",
]


def post_audit():
    log("=== post-pipeline audit (look for crash counts > 0) ===")
    for loc in AUDIT_DIRS:
        path = REPO / loc
        if not path.is_dir():
            continue
        n_total = n_crash = 0
        for fp in path.glob("*.json"):
            if "_crashed_" in str(fp):
                continue
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            n_total += 1
            if d.get("stop_reason") == "crash":
                n_crash += 1
        flag = "  <-- INVESTIGATE" if n_crash else ""
        log(f"  {loc:60s} total={n_total:5d}  crash={n_crash:4d}{flag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=float, default=50.0,
                    help="Stop after this many dollars are spent on THIS invocation (default 50)")
    ap.add_argument("--chunk-size", type=int, default=10,
                    help="Trajectory reroll chunk size (default 10)")
    ap.add_argument("--skip-quarantine", action="store_true",
                    help="Don't move crashed files (use on resume)")
    ap.add_argument("--skip-reroll", action="store_true",
                    help="Don't redo trajectory rollouts")
    ap.add_argument("--only-stages", default=None,
                    help="Comma-sep list of stage IDs (e.g. '3,4,5') — only run those pipeline stages")
    args = ap.parse_args()

    only = None
    if args.only_stages:
        only = {int(x) for x in args.only_stages.split(",")}

    log(f"=========================================================")
    log(f"recover.py START  run_id={RUN_ID}  cap=${args.cap:.2f}")
    log(f"  log file: {LOGFILE}")
    log(f"  initial: {fmt_cost()}")
    log(f"=========================================================")
    baseline = cost_snapshot()[2]

    # Stage 1 — quarantine
    if not args.skip_quarantine:
        log("\n[stage 1] quarantine crash-stop_reason files")
        moved = quarantine_crashes(RUN_ID)
        for loc, n in moved.items():
            log(f"  moved {n:4d}  {loc}")

    # Stage 2 — trajectory reroll
    if not args.skip_reroll:
        log("\n[stage 2] trajectory reroll")
        ids = crashed_trajectory_ids()
        log(f"  task IDs needing reroll: {len(ids)}")
        if ids:
            chunks = [ids[i:i + args.chunk_size]
                      for i in range(0, len(ids), args.chunk_size)]
            spend_floor_chunk0 = cost_snapshot()[2]
            for i, chunk in enumerate(chunks, 1):
                if stage_blocked(f"reroll-chunk-{i}/{len(chunks)}", baseline, args.cap):
                    break
                run_subproc(
                    f"reroll-chunk-{i:02d}/{len(chunks):02d} (n={len(chunk)})",
                    ["python", "-m", "agent.run_batch",
                     "--task-ids", *(str(t) for t in chunk),
                     "--max-steps", "50"],
                    baseline, args.cap,
                )
                if i == 1:
                    chunk0_delta = cost_snapshot()[2] - spend_floor_chunk0
                    if chunk0_delta < 1e-4:
                        log("ALERT: first reroll chunk produced ~$0 spend.")
                        log("  This usually means OpenRouter is still empty. Stopping early.")
                        log("  Top up the key, then re-run with --skip-quarantine.")
                        break

    # Stages 3-11 — pipeline
    log("\n[stages 3-11] pipeline")
    for stage_id, label, argv in PIPELINE:
        if only is not None and stage_id not in only:
            continue
        if stage_blocked(f"stage-{stage_id}-{label}", baseline, args.cap):
            log("Cap reached. Re-run with same args (or --skip-quarantine) to continue.")
            break
        run_subproc(f"stage-{stage_id}-{label}", argv, baseline, args.cap)

    # Final summary
    log("\n=========================================================")
    log("recover.py FINISHED")
    final = cost_snapshot()[2]
    log(f"  initial total:  ${baseline:.4f}")
    log(f"  final total:    ${final:.4f}")
    log(f"  spent this run: ${final - baseline:.4f}  (cap ${args.cap:.2f})")
    log(f"  log: {LOGFILE}")
    log("=========================================================")
    post_audit()


if __name__ == "__main__":
    sys.exit(main() or 0)
