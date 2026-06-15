#!/usr/bin/env python3
"""STN-FOMO: measure the actual learned rotation angle per image.

For every image in the test set, runs the (fp32) model with the localization
network's raw output exposed -- the `linear_1` tensor, shape (1,2),
(t1, t2) ~= (cos a, sin a) -- and converts it to the rotation angle

    alpha = atan2(t2, t1)   (degrees, range (-180, 180])

that the STN actually applied to that image's feature map. Also reports the
vector magnitude sqrt(t1^2 + t2^2), which should be ~1 if the network has
learned a pure rotation (no implicit scaling).

For images with a railroad-crossing (class 0) annotation, alpha is compared
against the gate's own OBB tilt (folded to [0,90], 0=horizontal/closed,
90=vertical/open) to test the "canonical pose" hypothesis from the tilt-bin
ablation:

  - alpha strongly anti-correlated with tilt          -> STN "undoes" each
      image's own rotation toward a fixed reference frame.
  - |alpha| correlated with distance-from-canonical
      (= min(tilt, 90-tilt), max at tilt=45)           -> STN applies bigger
      corrections the further the gate is from horizontal/vertical.
  - alpha ~ 0 everywhere, no correlation               -> STN has converged to
      a near-identity transform; geometric normalization is largely inert for
      this dataset (consistent with ~0deg findings from the orientation/tilt-
      bin experiments).

Outputs
-------
  stdout                                  - overall stats, per-tilt-bin table,
                                            correlations, insights
  <output>/learned_angle_<slug>.csv       - per-image: has_gate, tilt_deg,
                                            t1, t2, alpha_deg, magnitude
  <output>/learned_angle_<slug>.png       - 2-panel figure:
                                            (1) scatter: gate tilt vs learned alpha
                                                (+ per-bin mean +/- std)
                                            (2) histogram of learned alpha (all images)

Usage
-----
    python3 evaluate-stn-learned-angle.py
    python3 evaluate-stn-learned-angle.py --images path/to/img --labels path/to/lbl
    python3 evaluate-stn-learned-angle.py --bins 0 10 30 60 80 90
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "fp32": str(SCRIPT_DIR / "models/stn-fomo-480-onnx/stn-fomo-480.onnx"),
    "int8": str(SCRIPT_DIR / "models/stn-fomo-480-onnx/stn-fomo-480-int8.onnx"),
}
DEFAULT_IMAGES  = str(SCRIPT_DIR / "datasets/yolo_pl_test/images")
DEFAULT_LABELS  = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_OUTPUT  = str(SCRIPT_DIR / "benchmark_results/stn-learned-angle")

# Tilt bins (gate angle from horizontal, folded to [0,90])
DEFAULT_BINS = [0, 10, 30, 60, 80, 90]
# Localization-network output tensor (fp32 graph): shape (1,2) ~= (cos a, sin a)
DEFAULT_LOC_TENSOR = "linear_1"


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------
def _load_eval_module():
    path = SCRIPT_DIR / "evaluate-fomo.py"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find evaluate-fomo.py at {path}")
    spec = importlib.util.spec_from_file_location("evaluate_fomo", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _add_localization_output(model: onnx.ModelProto, tensor_name: str) -> bool:
    """Append `tensor_name` (the (1,2) localization-net output) as a graph output."""
    produced = set()
    for n in model.graph.node:
        produced.update(n.output)
    if tensor_name not in produced:
        return False
    if any(o.name == tensor_name for o in model.graph.output):
        return True
    vi = helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, [1, 2])
    model.graph.output.append(vi)
    return True


def _make_session(model_path: str, providers: list[str], num_threads: int,
                   loc_tensor: str) -> tuple[ort.InferenceSession, int]:
    """Build a session with `loc_tensor` exposed as an additional output.

    Raises RuntimeError if the tensor is not present in the graph (e.g. an
    int8-quantized model where intermediate tensor names were rewritten).
    """
    m = onnx.load(model_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass

    if not _add_localization_output(m, loc_tensor):
        raise RuntimeError(
            f"Tensor '{loc_tensor}' not found in {model_path}. "
            f"This script requires the fp32 graph with the standard STN "
            f"localization head (loc_fc.0 -> ReLU -> loc_fc.2 -> '{loc_tensor}')."
        )

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    configured = []
    for p in providers:
        if p == "XnnpackExecutionProvider":
            configured.append((p, {"intra_op_num_threads": str(num_threads)}))
        else:
            configured.append(p)

    sess = ort.InferenceSession(m.SerializeToString(), sess_options=opts, providers=configured)
    try:
        input_size = int(sess.get_inputs()[0].shape[2])
    except (TypeError, ValueError):
        input_size = 480
    return sess, input_size


# ---------------------------------------------------------------------------
# Tilt angle from OBB label  (mirrors evaluate-stn-tilt-bins.py)
# ---------------------------------------------------------------------------
def _gate_tilt(label_path: Path) -> float | None:
    """Return the tilt-from-horizontal of the railroad-crossing OBB, or None.

    Tilt is in [0,90]: 0=horizontal, 90=vertical.
    If multiple class-0 annotations exist, returns the maximum tilt.
    """
    tilts = []
    try:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9 or parts[0] != "0":
                continue
            coords = list(map(float, parts[1:9]))
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            tilt  = min(angle, 180 - angle)   # fold to [0,90]
            tilts.append(tilt)
    except (FileNotFoundError, ValueError):
        pass
    return max(tilts) if tilts else None


def _bin_index(tilt: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        if tilt < hi or i == len(edges) - 2:
            return i
    return len(edges) - 2


def _bin_label(edges: list[float], i: int) -> str:
    return f"{edges[i]:.0f}–{edges[i+1]:.0f}°"


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_overall(alpha: np.ndarray, magnitude: np.ndarray):
    print("\n" + "─" * 72)
    print("  OVERALL  (all images)")
    print("─" * 72)
    print(f"  n images               : {len(alpha)}")
    print(f"  alpha   mean / std      : {alpha.mean():+.3f}° / {alpha.std():.3f}°")
    print(f"  alpha   min / max       : {alpha.min():+.3f}° / {alpha.max():+.3f}°")
    print(f"  |alpha| mean / std      : {np.abs(alpha).mean():.3f}° / {np.abs(alpha).std():.3f}°")
    print(f"  magnitude mean / std    : {magnitude.mean():.4f} / {magnitude.std():.4f}  "
          f"(1.0 == pure rotation, no scaling)")
    print("─" * 72)


def print_bin_table(bin_names: list[str], bin_stats: list[dict | None]):
    print("\n" + "─" * 96)
    print(f"  {'Tilt bin':<10}  {'n':>5}  {'mean tilt':>10}  {'mean alpha':>11}  "
          f"{'std alpha':>10}  {'mean |alpha|':>13}  {'mean mag':>9}")
    print(f"  alpha = atan2(t2,t1) learned per image  ·  railroad-crossing gates only")
    print("─" * 96)
    for name, bs in zip(bin_names, bin_stats):
        if bs is None:
            print(f"  {name:<10}  {0:>5}   (no samples)")
            continue
        print(
            f"  {name:<10}  {bs['n']:>5}  {bs['tilt_mean']:>9.2f}°  "
            f"{bs['alpha_mean']:>+10.3f}°  {bs['alpha_std']:>9.3f}°  "
            f"{bs['abs_alpha_mean']:>12.3f}°  {bs['mag_mean']:>9.4f}"
        )
    print("─" * 96)


def print_insights(bin_names: list[str], bin_stats: list[dict | None],
                    r_tilt_alpha: float, r_tilt_absalpha: float, r_dist_absalpha: float,
                    overall_abs_alpha_mean: float):
    print("\n" + "─" * 72)
    print("  INSIGHTS")
    print("─" * 72)

    print(f"\n  Correlations (railroad-crossing gates only):")
    print(f"    corr(tilt, alpha)                    = {r_tilt_alpha:+.3f}")
    print(f"    corr(tilt, |alpha|)                  = {r_tilt_absalpha:+.3f}")
    print(f"    corr(dist-from-canonical, |alpha|)   = {r_dist_absalpha:+.3f}")
    print(f"    (dist-from-canonical = min(tilt, 90-tilt), max at tilt=45°)")

    if overall_abs_alpha_mean < 1.0:
        print(f"\n  Mean |alpha| across all images is only {overall_abs_alpha_mean:.3f}°.")
        print(f"  -> The STN has converged to a near-identity rotation; its output is")
        print(f"     essentially constant regardless of the gate's actual tilt. The")
        print(f"     geometric-normalization branch is largely inert on this dataset.")
    elif abs(r_tilt_alpha) > 0.4:
        print(f"\n  alpha correlates with tilt (r={r_tilt_alpha:+.3f}).")
        print(f"  -> Consistent with the STN 'undoing' each image's own rotation toward")
        print(f"     a fixed reference frame (rotation-normalization behaviour).")
    elif r_dist_absalpha > 0.4:
        print(f"\n  |alpha| correlates with distance-from-canonical (r={r_dist_absalpha:+.3f}).")
        print(f"  -> Consistent with the STN applying larger corrections to gates that")
        print(f"     are furthest from a horizontal/vertical (canonical) pose.")
    else:
        print(f"\n  No strong correlation found between learned alpha and gate tilt.")

    # Per-bin bar of mean |alpha|
    print(f"\n  mean |alpha| per tilt bin:")
    for name, bs in zip(bin_names, bin_stats):
        if bs is None:
            continue
        bar_len = int(bs["abs_alpha_mean"] * 10)
        bar = "▓" * min(bar_len, 40)
        print(f"    {name:<10}  {bs['abs_alpha_mean']:>6.3f}°  {bar}")

    print("─" * 72)


def save_csv(path: str, rows: list[dict]):
    if not rows:
        print("  (no rows to write)")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] CSV -> {path}")


def save_plot(path: str, bin_names: list[str], bin_stats: list[dict | None],
               tilts: np.ndarray, alphas_gate: np.ndarray, alpha_all: np.ndarray,
               precision: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available - skipping plot)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # -- Panel 1: scatter tilt vs alpha, + per-bin mean +/- std -------------
    ax = axes[0]
    ax.scatter(tilts, alphas_gate, s=14, alpha=0.35, color="#2563EB",
               label="per-image alpha")

    bin_centers, bin_means, bin_stds = [], [], []
    for name, bs in zip(bin_names, bin_stats):
        if bs is None:
            continue
        bin_centers.append(bs["tilt_mean"])
        bin_means.append(bs["alpha_mean"])
        bin_stds.append(bs["alpha_std"])
    if bin_centers:
        ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt="o-",
                     color="#DC2626", linewidth=1.8, markersize=6, capsize=4,
                     label="per-bin mean +/- std")

    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
               label="alpha = 0 (identity)")
    ax.set_xlabel("gate tilt (deg, 0=horizontal, 90=vertical)")
    ax.set_ylabel("learned rotation alpha (deg)")
    ax.set_title("Learned STN rotation vs gate tilt", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # -- Panel 2: histogram of alpha (all images) ----------------------------
    ax = axes[1]
    ax.hist(alpha_all, bins=40, color="#16A34A", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.axvline(alpha_all.mean(), color="#DC2626", linewidth=1.5, linestyle="-",
               label=f"mean = {alpha_all.mean():+.3f}°")
    ax.set_xlabel("learned rotation alpha (deg)")
    ax.set_ylabel("# images")
    ax.set_title("Distribution of learned alpha (all images)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"STN-FOMO {precision.upper()} - learned rotation angle per image",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] plot -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--bins", nargs="+", type=float, default=DEFAULT_BINS,
                    help="Tilt bin edges in degrees (default: 0 10 30 60 80 90).")
    ap.add_argument("--loc-tensor", default=DEFAULT_LOC_TENSOR,
                    help=f"Localization-net output tensor name (default: {DEFAULT_LOC_TENSOR}).")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    bin_edges = sorted(args.bins)
    if len(bin_edges) < 2:
        raise SystemExit("--bins needs at least 2 edge values")

    model_path = MODEL_PATHS.get(args.precision)
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    os.makedirs(args.output, exist_ok=True)
    ev = _load_eval_module()
    n_bins = len(bin_edges) - 1
    bin_names = [_bin_label(bin_edges, i) for i in range(n_bins)]

    print(f"\nSTN-FOMO learned-angle measurement  ·  {args.precision.upper()}")
    print(f"Tilt bins: {' | '.join(bin_names)}")
    print(f"Images: {args.images}")

    print("\nLoading session (with localization output exposed) …")
    sess, input_size = _make_session(
        model_path, args.providers, args.threads, args.loc_tensor,
    )
    in_name  = sess.get_inputs()[0].name
    out_name = args.loc_tensor

    img_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in ev.IMG_EXTS
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    rows = []
    total = len(img_paths)
    for done, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = Path(args.labels) / (img_path.stem + ".txt")
        tilt = _gate_tilt(label_path)

        blob = ev.preprocess(img, input_size)
        t1, t2 = sess.run([out_name], {in_name: blob})[0][0]
        t1, t2 = float(t1), float(t2)
        alpha = math.degrees(math.atan2(t2, t1))
        magnitude = math.hypot(t1, t2)

        rows.append({
            "image":     img_path.name,
            "has_gate":  tilt is not None,
            "tilt_deg":  "" if tilt is None else f"{tilt:.3f}",
            "t1":        f"{t1:.6f}",
            "t2":        f"{t2:.6f}",
            "alpha_deg": f"{alpha:.4f}",
            "magnitude": f"{magnitude:.5f}",
        })

        if done % 50 == 0 or done == total:
            print(f"  [{done:>4}/{total}]")

    # -------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------
    alpha_all = np.array([float(r["alpha_deg"]) for r in rows])
    mag_all   = np.array([float(r["magnitude"]) for r in rows])

    gate_rows = [r for r in rows if r["has_gate"]]
    tilts        = np.array([float(r["tilt_deg"]) for r in gate_rows])
    alphas_gate  = np.array([float(r["alpha_deg"]) for r in gate_rows])
    mags_gate    = np.array([float(r["magnitude"]) for r in gate_rows])
    abs_alphas   = np.abs(alphas_gate)
    dist_canon   = np.minimum(tilts, 90.0 - tilts)

    bin_idx = np.array([_bin_index(t, bin_edges) for t in tilts]) if len(tilts) else np.array([])

    bin_stats: list[dict | None] = []
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            bin_stats.append(None)
            continue
        bin_stats.append({
            "n":              int(mask.sum()),
            "tilt_mean":      float(tilts[mask].mean()),
            "alpha_mean":     float(alphas_gate[mask].mean()),
            "alpha_std":      float(alphas_gate[mask].std()),
            "abs_alpha_mean": float(abs_alphas[mask].mean()),
            "mag_mean":       float(mags_gate[mask].mean()),
        })

    r_tilt_alpha    = _pearson(tilts, alphas_gate)
    r_tilt_absalpha = _pearson(tilts, abs_alphas)
    r_dist_absalpha = _pearson(dist_canon, abs_alphas)

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    print_overall(alpha_all, mag_all)
    print_bin_table(bin_names, bin_stats)
    print_insights(bin_names, bin_stats, r_tilt_alpha, r_tilt_absalpha, r_dist_absalpha,
                    float(np.abs(alpha_all).mean()))

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
    slug = f"stn-fomo-{args.precision}"
    save_csv(os.path.join(args.output, f"learned_angle_{slug}.csv"), rows)
    save_plot(
        os.path.join(args.output, f"learned_angle_{slug}.png"),
        bin_names, bin_stats, tilts, alphas_gate, alpha_all, args.precision,
    )
    print(f"\nArtifacts written to: {args.output}")


if __name__ == "__main__":
    main()
