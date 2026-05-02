"""Compute the human-attribution rate alpha_hat_human_context (Proposal Section 2.3).

For every failed-trajectory annotation, count how many annotators classified
the failure as 'context-caused' vs 'capability-caused'. The rate is reported
per-annotator and pooled, so it can be compared against alpha_context from
the oracle intervention (compute_alpha.py) for the convergent-validity check.

Annotation schema (one file per task per annotator):
    annotations/<annotator>/task_<id>.json
        failure_classification: "context-caused" | "capability-caused" | other

Outputs experiments/alpha_human_results.json.
"""
import json
import os
from collections import Counter, defaultdict

ANNOTATION_ROOT = "annotations"
OUT_PATH = "experiments/alpha_human_results.json"


def load_annotations():
    """{annotator: {task_id: classification}}, normalized to lowercase."""
    out = defaultdict(dict)
    if not os.path.isdir(ANNOTATION_ROOT):
        return out
    for ann in sorted(os.listdir(ANNOTATION_ROOT)):
        ann_dir = os.path.join(ANNOTATION_ROOT, ann)
        if not os.path.isdir(ann_dir) or not ann.startswith("annotator_"):
            continue
        for fn in os.listdir(ann_dir):
            if not (fn.startswith("task_") and fn.endswith(".json")):
                continue
            try:
                d = json.load(open(os.path.join(ann_dir, fn)))
            except Exception:
                continue
            cls = (d.get("failure_classification") or "").strip().lower()
            tid = fn.replace("task_", "").replace(".json", "")
            if cls:
                out[ann][tid] = cls
    return out


def rate(counts):
    n = sum(counts.values())
    if n == 0:
        return None, 0
    n_ctx = counts.get("context-caused", 0)
    return n_ctx / n, n


def main():
    anns = load_annotations()
    if not anns:
        print("No annotations found.")
        return

    out = {"per_annotator": {}}
    pooled = Counter()
    union_tasks = set()
    for ann, mapping in anns.items():
        c = Counter(mapping.values())
        r, n = rate(c)
        out["per_annotator"][ann] = {
            "n": n,
            "alpha_hat_context": round(r, 4) if r is not None else None,
            "class_counts": dict(c),
        }
        pooled.update(c)
        union_tasks.update(mapping.keys())

    r_p, n_p = rate(pooled)
    out["pooled_all_annotators"] = {
        "n_annotations": n_p,
        "n_distinct_tasks": len(union_tasks),
        "alpha_hat_context": round(r_p, 4) if r_p is not None else None,
        "class_counts": dict(pooled),
        "note": "Pooled treats each annotator-label as one observation. "
                "Tasks labeled by multiple annotators contribute multiple times. "
                "For a per-task vote, use 'per_task_majority' below.",
    }

    # Per-task majority vote: each task gets one verdict (ties broken as 'tie')
    per_task = Counter()
    tie_tasks = []
    for tid in union_tasks:
        votes = Counter()
        for ann, mapping in anns.items():
            if tid in mapping:
                votes[mapping[tid]] += 1
        # majority
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            per_task["tie"] += 1
            tie_tasks.append(tid)
        else:
            per_task[top[0][0]] += 1
    n_total = sum(per_task.values())
    n_ctx = per_task.get("context-caused", 0)
    n_cap = per_task.get("capability-caused", 0)
    out["per_task_majority"] = {
        "n_tasks": n_total,
        "context_caused":   n_ctx,
        "capability_caused": n_cap,
        "tie": per_task.get("tie", 0),
        "alpha_hat_context": round(n_ctx / n_total, 4) if n_total else None,
        "tie_tasks": sorted(tie_tasks),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
