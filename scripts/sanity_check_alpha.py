"""Sanity-check the n=12 clean intervention pairs vs the 212 crashed pairs.

If the 12 surviving pairs are systematically different from the 212 crashed,
alpha_results.json is selection-biased and needs reweighting or more data.
If they look similar, the directional finding (capability dominates) is robust
to selection.

No API calls. Reads only local files. Run anywhere.

    python scripts/sanity_check_alpha.py
"""
import json, os, glob
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASELINE = REPO / "experiments/oracle_baseline"
RAW = REPO / "trajectories/data"


def site_of(meta) -> str:
    sites = meta.get("sites")
    if isinstance(sites, list) and sites:
        return "+".join(sites) if len(sites) > 1 else sites[0]
    return "unknown"


def load_pairs():
    """Return {(tid, ann): {'oracle': result, 'envonly': result}}."""
    pairs = defaultdict(dict)
    for f in glob.glob(str(BASELINE / "*.json")):
        n = os.path.basename(f).replace("_result.json", "")
        parts = n.split("_")
        if len(parts) < 4 or parts[-2] != "tstar":
            continue
        cond = parts[-1]
        if cond not in ("oracle", "envonly"):
            continue
        tid = parts[0]
        ann = "_".join(parts[1:-2]) or "legacy"
        try:
            pairs[(tid, ann)][cond] = json.load(open(f))
        except Exception:
            pass
    return pairs


def load_raw_meta():
    """Return {tid: {'site': ..., 'n_steps': ..., 'success': ..., 'stop_reason': ...}}."""
    out = {}
    for f in glob.glob(str(RAW / "*.json")):
        tid = os.path.basename(f).replace(".json", "")
        try:
            d = json.load(open(f))
            out[tid] = {
                "site": site_of(d),
                "n_steps": len(d.get("steps") or []),
                "success": d.get("success"),
                "stop_reason": d.get("stop_reason"),
            }
        except Exception:
            pass
    return out


def summarise(label, tids, raw_meta):
    sites = Counter()
    lens = []
    stops = Counter()
    succ = Counter()
    for tid in tids:
        m = raw_meta.get(tid)
        if not m:
            sites["NO_RAW"] += 1
            continue
        sites[m["site"]] += 1
        lens.append(m["n_steps"])
        stops[str(m["stop_reason"])] += 1
        succ[str(m["success"])] += 1
    avg_len = sum(lens) / len(lens) if lens else 0
    print(f"\n--- {label} (n_tasks={len(tids)}) ---")
    print(f"  sites:        {dict(sites)}")
    print(f"  avg_steps:    {avg_len:.1f}  (min {min(lens) if lens else 0}, max {max(lens) if lens else 0})")
    print(f"  raw success:  {dict(succ)}")
    print(f"  raw stop:     {dict(stops)}")


def main():
    pairs = load_pairs()
    raw = load_raw_meta()
    print(f"loaded {len(pairs)} (task, annotator) intervention pairs")
    print(f"loaded {len(raw)} raw trajectories")

    clean_tasks = set()
    crashed_tasks = set()
    for (tid, ann), v in pairs.items():
        if "oracle" not in v or "envonly" not in v:
            continue
        s_o = v["oracle"].get("success")
        s_e = v["envonly"].get("success")
        if s_o is None or s_e is None:
            crashed_tasks.add(tid)
        else:
            clean_tasks.add(tid)

    overlap = clean_tasks & crashed_tasks
    print(f"\nunique tasks with at least one CLEAN pair:   {len(clean_tasks)}")
    print(f"unique tasks with ANY crashed pair:           {len(crashed_tasks)}")
    print(f"overlap (mixed clean+crashed annotations):    {len(overlap)}")

    summarise("CLEAN-only tasks (drove the n=12 alpha)",
              clean_tasks - crashed_tasks, raw)
    summarise("CRASHED-only tasks (excluded from alpha)",
              crashed_tasks - clean_tasks, raw)
    summarise("Mixed tasks (some clean, some crashed)",
              overlap, raw)

    # Per-pair breakdown for the actual alpha-contributing pairs
    print("\n--- Per-pair (task, annotator) outcomes for the 12 clean pairs ---")
    n = 0
    for (tid, ann), v in sorted(pairs.items()):
        if "oracle" not in v or "envonly" not in v:
            continue
        s_o = v["oracle"].get("success")
        s_e = v["envonly"].get("success")
        if s_o is None or s_e is None:
            continue
        m = raw.get(tid, {})
        site = m.get("site", "?")
        steps = m.get("n_steps", "?")
        print(f"  task={tid:>4s}  ann={ann:<14s}  site={site:<14s}  raw_steps={steps:>3}  "
              f"oracle_success={s_o}  envonly_success={s_e}")
        n += 1
    print(f"\ntotal clean pairs contributing to alpha: {n}")


if __name__ == "__main__":
    main()
