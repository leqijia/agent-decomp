"""Plot raw-condition success rate by trajectory length bin with 95% Wilson CIs.

Usage:
    python scripts/plot_raw_acc_by_length.py
    python scripts/plot_raw_acc_by_length.py \\
        --in experiments/exp3/raw/metrics.json \\
        --out paper/figures/raw_acc_by_length.png \\
        --title-suffix " (draft)"
"""
import argparse
import json
import os
import sys

BIN_ORDER  = ["<=10", "11-20", "21-40", "41-80"]
BIN_LABELS = ["≤10",  "11–20", "21–40", "41–80"]


def main():
    parser = argparse.ArgumentParser(
        description="Plot raw-condition success rate by trajectory length bin"
    )
    parser.add_argument("--in", dest="in_path",
                        default="experiments/exp3/raw/metrics.json")
    parser.add_argument("--out", default="paper/figures/raw_acc_by_length.png")
    parser.add_argument("--title-suffix", dest="title_suffix", default="")
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print(f"ERROR: input file not found: {args.in_path}", file=sys.stderr)
        sys.exit(1)

    with open(args.in_path) as f:
        data = json.load(f)

    if not data:
        print(f"ERROR: input file is empty: {args.in_path}", file=sys.stderr)
        sys.exit(1)

    by_bin = data.get("by_length_bin", {})
    extra = set(by_bin.keys()) - set(BIN_ORDER)
    if extra:
        print(f"  WARNING: unexpected bin keys {extra} — rendering canonical bins only")

    present = [b for b in BIN_ORDER if b in by_bin]
    missing = [b for b in BIN_ORDER if b not in by_bin]
    if missing:
        print(f"  WARNING: missing bin keys {missing} — those bins will be absent")

    rates, ns, err_lo, err_hi = [], [], [], []
    for b in present:
        entry = by_bin[b]
        sr   = entry["success_rate"]
        ci   = entry["ci_95"]
        n    = entry["n"]
        rates.append(sr)
        ns.append(n)
        err_lo.append(sr - ci[0])
        err_hi.append(ci[1] - sr)

    labels = [BIN_LABELS[BIN_ORDER.index(b)] for b in present]
    overall = data.get("overall", {})
    overall_sr = overall.get("success_rate")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"ERROR: matplotlib and numpy are required — {e}", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(present))

    bars = ax.bar(x, rates, color="#888888", zorder=2)
    ax.errorbar(
        x, rates,
        yerr=[err_lo, err_hi],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3,
    )

    # Annotate n= above each CI upper bound
    for i, (sr, n, eh) in enumerate(zip(rates, ns, err_hi)):
        top = sr + eh
        ax.text(x[i], top + 0.012, f"n={n}", ha="center", va="bottom",
                fontsize=9)

    # Overall dashed line
    if overall_sr is not None:
        ax.axhline(overall_sr, color="#333333", linestyle="--",
                   linewidth=1.2, zorder=1, label=f"overall ({overall_sr:.3f})")
        ax.legend(loc="upper right", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Trajectory length (steps)")
    ax.set_ylabel("Task success rate")
    ax.set_ylim(0, 0.5)
    ax.set_title(
        f"Raw condition: agent success vs trajectory length{args.title_suffix}"
    )
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    pdf_path = os.path.splitext(args.out)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved PNG: {args.out}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
