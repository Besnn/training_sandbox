#!/usr/bin/env python3
"""Overlay precision-recall curves for FOMO vs STN-FOMO, fp32 vs int8.

Same evaluation/decoding/matching as plot_pr_curves_fomo.py, but instead of one
PNG per model, draws a single figure with one subplot per class containing all
four model curves (FOMO-STN-FP32/INT8, FOMO-FP32/INT8) overlaid, each annotated
with AP and its best-F1 operating point.

Usage:
    python3 plot_pr_overlay.py
    python3 plot_pr_overlay.py --dist 0.05 --output benchmark_results/pr_overlay.png
"""

import argparse
import os

from plot_pr_curves_fomo import (
    DEFAULT_CLASSES,
    DEFAULT_MODELS,
    DEFAULT_VAL_IMAGES,
    DEFAULT_VAL_LABELS,
    PRIMARY_DISTANCE,
    CONF_FLOOR,
    SCRIPT_DIR,
    evaluate,
    resolve_model_path,
)
from plot_f1_comparison import COLORS

DEFAULT_OUTPUT = str(SCRIPT_DIR / "benchmark_results/pr-curves/pr_overlay.png")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=DEFAULT_VAL_IMAGES)
    ap.add_argument("--labels", default=DEFAULT_VAL_LABELS)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--dist", type=float, default=PRIMARY_DISTANCE,
                    help="Match distance in normalized image units.")
    ap.add_argument("--conf-floor", type=float, default=CONF_FLOOR,
                    help="Loose confidence floor for collecting detections.")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    return ap.parse_args()


def plot_overlay(results_by_model, classes, dist_thres, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(classes)
    cols = 2
    rows = (nc + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 5.2 * rows), squeeze=False)
    fig.suptitle(f"FOMO vs STN-FOMO — precision-recall (dist ≤ {dist_thres} norm)",
                 fontsize=15, fontweight="bold")

    model_names = [n for n in DEFAULT_MODELS if n in results_by_model]

    for c, cname in enumerate(classes):
        ax = axes[c // cols][c % cols]
        ax.set_title(cname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        any_data = False
        for name in model_names:
            r = results_by_model[name][c]
            color = COLORS.get(name)
            if r["recall"].size == 0:
                continue
            any_data = True
            ax.plot(r["recall"], r["precision"], color=color, lw=2,
                    label=f"{name} (AP={r['ap']:.3f}, F1={r['best_f1']:.3f})")
            ax.plot(r["best_r"], r["best_p"], "o", color=color, ms=7, zorder=5)

        if not any_data:
            ax.text(0.5, 0.5, "no data\n(no GT or no detections)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            continue
        ax.legend(loc="lower left", fontsize=8)

    for k in range(nc, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    classes = args.classes

    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    results_by_model = {}
    for name, mp in DEFAULT_MODELS.items():
        model_path = resolve_model_path(mp)
        if not os.path.isfile(model_path):
            print(f"[WARN] {name}: model not found at {mp} — skipping")
            continue
        print(f"\n=== {name} ({model_path}) ===")
        results = evaluate(model_path, args.images, args.labels, args.providers,
                           classes, args.dist, args.conf_floor)
        results_by_model[name] = results
        for c, cname in enumerate(classes):
            r = results[c]
            print(f"  {cname:<22} P={r['best_p']:.3f} R={r['best_r']:.3f} "
                  f"F1={r['best_f1']:.3f} AP={r['ap']:.3f}")

    if not results_by_model:
        raise SystemExit("No models evaluated.")

    plot_overlay(results_by_model, classes, args.dist, args.output)
    print(f"\n[OK] PR overlay written to {args.output}")


if __name__ == "__main__":
    main()
