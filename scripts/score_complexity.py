"""Score WebArena config tasks by complexity to find longest-trajectory candidates."""

import json
import glob
import os
import re
from collections import defaultdict

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config_files")
ALLOWED_SITES = {"shopping", "shopping_admin", "reddit", "gitlab"}

ALREADY_COLLECTED = {
    29, 62, 114, 130, 161, 192, 199, 231, 276, 282, 308, 322, 324, 327, 332,
    342, 349, 355, 387, 388, 401, 494, 529, 530, 532, 542, 552, 553, 554, 555,
    562, 563, 564, 565, 566, 571, 588, 599, 619, 630, 651, 657, 668, 671, 692,
    698, 730, 732, 796, 809,
}

MULTI_OP_KEYWORDS = [
    " and ", " then ", " also ", " after ", " next ", " finally ",
    " followed by ", " as well as ", " in addition ",
    " both ", " first ", " second ", " third ",
    " create ", " change ", " update ", " delete ", " remove ",
    " post ", " comment ", " reply ", " submit ",
]

def score_task(config: dict) -> dict:
    sites = config["sites"]
    task_id = config["task_id"]
    intent = config.get("intent", "")
    eval_info = config.get("eval", {})
    eval_types = eval_info.get("eval_types", [])
    program_html = eval_info.get("program_html", [])
    reference_url = eval_info.get("reference_url", "")

    # --- Filter: only allowed sites ---
    for s in sites:
        if s not in ALLOWED_SITES:
            return None

    score = 0.0

    # 1. Multi-site bonus (big signal)
    if len(sites) > 1:
        score += 25.0

    # 2. Eval type scoring
    has_program_html = "program_html" in eval_types
    has_url_match = "url_match" in eval_types
    has_string_match = "string_match" in eval_types

    if has_program_html and has_url_match:
        score += 15.0  # dual eval = action task verified at URL + DOM
    elif has_program_html:
        score += 10.0  # action task verified by DOM
    elif has_url_match:
        score += 8.0   # must navigate to correct URL
    elif has_string_match:
        score += 2.0   # info-lookup, simplest

    # 3. Number of program_html checks (more checks = more things to verify)
    if program_html:
        n_checks = len(program_html)
        score += min(n_checks * 3.0, 12.0)

    # 4. Intent length (proxy for number of steps)
    intent_words = len(intent.split())
    score += min(intent_words * 0.3, 15.0)  # cap at 15

    # 5. Multi-operation keywords in intent
    intent_lower = intent.lower()
    keyword_hits = sum(1 for kw in MULTI_OP_KEYWORDS if kw in intent_lower)
    score += min(keyword_hits * 2.0, 12.0)

    # 6. Comma-separated action lists in intent
    comma_count = intent.count(",")
    score += min(comma_count * 1.0, 6.0)

    # 7. Quoted strings in intent (each is a distinct piece of content to handle)
    quoted = re.findall(r'"[^"]*"', intent)
    score += min(len(quoted) * 1.5, 6.0)

    # 8. require_reset = True means state-modifying task
    if config.get("require_reset", False):
        score += 3.0

    # 9. Complex reference_url patterns (func: calls)
    if reference_url and "func:" in reference_url:
        score += 3.0
    if program_html:
        for ph in program_html:
            if isinstance(ph, dict) and "func:" in ph.get("url", ""):
                score += 2.0

    return {
        "task_id": task_id,
        "score": round(score, 1),
        "sites": sites,
        "eval_types": eval_types,
        "intent_preview": intent[:100],
        "intent_words": intent_words,
        "n_program_checks": len(program_html) if program_html else 0,
        "multi_site": len(sites) > 1,
    }


def main():
    files = glob.glob(os.path.join(CONFIG_DIR, "*.json"))
    results = []
    skipped_sites = 0
    skipped_collected = 0

    for f in files:
        with open(f) as fh:
            config = json.load(fh)

        task_id = config["task_id"]
        if task_id in ALREADY_COLLECTED:
            skipped_collected += 1
            continue

        scored = score_task(config)
        if scored is None:
            skipped_sites += 1
            continue

        results.append(scored)

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"Total configs: {len(files)}")
    print(f"Skipped (disallowed sites): {skipped_sites}")
    print(f"Skipped (already collected): {skipped_collected}")
    print(f"Scored: {len(results)}")
    print()

    top80 = results[:80]

    # Group by site combination
    groups = defaultdict(list)
    for r in top80:
        key = " + ".join(sorted(r["sites"]))
        groups[key].append(r)

    print("=" * 120)
    print(f"TOP 80 MOST COMPLEX TASKS (by site group)")
    print("=" * 120)

    for site_combo in sorted(groups.keys()):
        tasks = groups[site_combo]
        print(f"\n--- {site_combo} ({len(tasks)} tasks) ---")
        print(f"{'Rank':>4}  {'ID':>4}  {'Score':>6}  {'EvalTypes':<30}  {'Words':>5}  {'Checks':>6}  Intent Preview")
        print("-" * 120)
        for t in sorted(tasks, key=lambda x: x["score"], reverse=True):
            rank = top80.index(t) + 1
            et = ", ".join(t["eval_types"])
            print(f"{rank:>4}  {t['task_id']:>4}  {t['score']:>6}  {et:<30}  {t['intent_words']:>5}  {t['n_program_checks']:>6}  {t['intent_preview']}")

    print("\n" + "=" * 120)
    print("FLAT RANKED LIST (for easy copy)")
    print("=" * 120)
    print(f"{'Rank':>4}  {'ID':>4}  {'Score':>6}  {'Sites':<25}  {'EvalTypes':<30}  Intent Preview")
    print("-" * 120)
    for i, t in enumerate(top80):
        sites_str = " + ".join(sorted(t["sites"]))
        et = ", ".join(t["eval_types"])
        print(f"{i+1:>4}  {t['task_id']:>4}  {t['score']:>6}  {sites_str:<25}  {et:<30}  {t['intent_preview']}")

    # Summary stats
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    scores = [t["score"] for t in top80]
    print(f"Score range: {min(scores)} - {max(scores)}")
    print(f"Mean score: {sum(scores)/len(scores):.1f}")
    for site_combo in sorted(groups.keys()):
        print(f"  {site_combo}: {len(groups[site_combo])} tasks")

    # Task ID list for easy use
    ids = [t["task_id"] for t in top80]
    print(f"\nTask IDs (Python list):\n{ids}")


if __name__ == "__main__":
    main()
