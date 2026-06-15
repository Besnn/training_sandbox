#!/usr/bin/env python3
"""Plot best-F1 per class for FOMO vs STN-FOMO, fp32 vs int8.

Reuses the evaluation/decoding/matching logic from plot_pr_curves_fomo.py (same
decoding, same greedy matching, same best-F1-on-the-PR-curve definition) and
draws one grouped bar chart: one group per class (+ a "mean" group), with one
bar per model variant (FOMO-FP32, FOMO-INT8, FOMO-STN-FP32, FOMO-STN-INT8).

Usage:
    python3 plot_f1_comparison.py
    python3 plot_f1_comparison.py --dist 0.05 --output benchmark_results/f1.png
"""

import argparse
import os

import numpy as np

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

DEFAULT_OUTPUT = str(SCRIPT_DIR / "benchmark_results/pr-curves/f1_comparison.png")

COLORS = {
    "FOMO-STN-FP32": "#1F4E79",
    "FOMO-STN-INT8": "#7FB3E8",
    "FOMO-FP32": "#D7263D",
    "FOMO-INT8": "#F4A6AE",
}


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


def plot_f1(f1_by_model, classes, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(classes)
    labels = list(classes) + ["mean"]
    n_groups = len(labels)
    model_names = [n for n in DEFAULT_MODELS if n in f1_by_model]
    n_models = len(model_names)
    width = 0.8 / max(n_models, 1)

    x = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(2.0 * n_groups + 2, 5.5))
    for i, name in enumerate(model_names):
        per_class = [f1_by_model[name][c] for c in range(nc)]
        values = per_class + [float(np.mean(per_class))]
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=name, color=COLORS.get(name))
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best F1")
    ax.set_ylim(0, 1.08)
    ax.set_title("FOMO vs STN-FOMO — best F1 per class (fp32 vs int8)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    classes = args.classes
    nc = len(classes)

    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    f1_by_model = {}
    for name, mp in DEFAULT_MODELS.items():
        model_path = resolve_model_path(mp)
        if not os.path.isfile(model_path):
            print(f"[WARN] {name}: model not found at {mp} — skipping")
            continue
        print(f"\n=== {name} ({model_path}) ===")
        results = evaluate(model_path, args.images, args.labels, args.providers,
                           classes, args.dist, args.conf_floor)
        f1_by_model[name] = {c: results[c]["best_f1"] for c in range(nc)}
        for c, cname in enumerate(classes):
            print(f"  {cname:<22} F1={f1_by_model[name][c]:.3f}")

    if not f1_by_model:
        raise SystemExit("No models evaluated.")

    plot_f1(f1_by_model, classes, args.output)
    print(f"\n[OK] F1 comparison written to {args.output}")


if __name__ == "__main__":
    main()
