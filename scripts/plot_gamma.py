"""Plot Gamma(L) — two-panel publication figure from gamma_results.json.

Usage:
    python scripts/plot_gamma.py
    python scripts/plot_gamma.py --in experiments/gamma_results.json \\
        --out paper/figures/gamma_curve.png --title-suffix " (draft)"
"""
import argparse
import json
import os
import sys

BIN_ORDER = ["0-10", "11-20", "21-40", "41-80"]
BIN_LABELS = ["≤10", "11-20", "21-40", "41-80"]  # ≤10, 11-20, 21-40, 41-80


def _get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def parse_bin(d):
    acc_oracle = _get(d, "acc_oracle", "oracle_acc")
    acc_raw    = _get(d, "acc_raw",    "raw_acc")
    gamma      = _get(d, "gamma")
    n = _get(d, "n") or (
        max(_get(d, "oracle_n", default=0), _get(d, "raw_n", default=0)) or None
    )
    return acc_oracle, acc_raw, gamma, n


def main():
    parser = argparse.ArgumentParser(description="Plot Gamma(L) context gap figure")
    parser.add_argument("--in", dest="in_path",
                        default="experiments/gamma_results.json")
    parser.add_argument("--out", default="paper/figures/gamma_curve.png")
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

    extra = set(data.keys()) - set(BIN_ORDER) - {"80+"}
    if extra:
        print(f"  WARNING: unexpected bin keys {extra} — rendering canonical bins only")

    present_bins   = [b for b in BIN_ORDER if b in data]
    present_labels = [BIN_LABELS[BIN_ORDER.index(b)] for b in present_bins]
    missing = [b for b in BIN_ORDER if b not in data]
    if missing:
        print(f"  WARNING: missing bin keys {missing} — those bins will be absent")

    accs_oracle, accs_raw, gammas, ns = [], [], [], []
    for b in present_bins:
        ao, ar, g, n = parse_bin(data[b])
        accs_oracle.append(ao)
        accs_raw.append(ar)
        gammas.append(g)
        ns.append(n)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"ERROR: matplotlib and numpy are required — {e}", file=sys.stderr)
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    x = np.arange(len(present_bins))
    w = 0.38

    # ── Panel 1: paired bars ──────────────────────────────────────────────────
    ax1.bar(x - w / 2, accs_raw,    w, color="#888888", label="Raw")
    ax1.bar(x + w / 2, accs_oracle, w, color="#1f77b4", label="Oracle")
    ax1.set_xticks(x)
    ax1.set_xticklabels(present_labels)
    ax1.set_xlabel("Trajectory length (steps)")
    ax1.set_ylabel("Task success rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("Accuracy by trajectory length")
    ax1.legend(loc="upper right")
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)
    for i, n in enumerate(ns):
        if n is not None:
            ax1.text(x[i], -0.07, f"n={n}", ha="center", va="top",
                     fontsize=8, transform=ax1.get_xaxis_transform())

    # ── Panel 2: gamma line ───────────────────────────────────────────────────
    ax2.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax2.plot(x, gammas, color="#d62728", marker="o",
             linewidth=2, markersize=8, zorder=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(present_labels)
    ax2.set_xlabel("Trajectory length (steps)")
    ax2.set_ylabel(r"$\Gamma(L)$ = Acc(oracle) $-$ Acc(raw)")
    ax2.set_title(r"Context gap $\Gamma(L)$")
    ax2.yaxis.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    for i, (g, n) in enumerate(zip(gammas, ns)):
        if g is None:
            continue
        sign = "+" if g >= 0 else ""
        label = f"{sign}{g:.2f}"
        if n is not None:
            label += f"\n(n={n})"
        va     = "bottom" if g >= 0 else "top"
        offset = 0.01    if g >= 0 else -0.01
        ax2.annotate(label, (x[i], g + offset), ha="center", va=va, fontsize=8)

    fig.suptitle(
        f"Experiment 2: Performance-length scaling{args.title_suffix}",
        fontsize=12,
    )
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
