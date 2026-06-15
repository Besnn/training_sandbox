#!/usr/bin/env python3
"""STN-FOMO ablation: compare detection quality with and without the STN warp.

Loads STN-FOMO and evaluates it twice on the same test set:

  STN-active  — normal model, the learned spatial transformer is applied
  STN-bypass  — identity grid replaces the warp  (θ = [[1,0,0],[0,1,0]])
                backbone output flows directly to the detection head

All metric code is imported from evaluate-fomo.py so numbers are directly
comparable to the standard benchmarks (centroid L2 distance matching,
COCO-style distance-sweep AP, row-normalised confusion matrix).

Outputs
-------
  stdout      : side-by-side per-class table, delta table, insights paragraph
  <output-dir>/metrics_active.csv / metrics_bypass.csv
  <output-dir>/confusion_active.png / confusion_bypass.png
  <output-dir>/comparison.png

Usage
-----
    python3 evaluate-stn-bypass.py
    python3 evaluate-stn-bypass.py --precision int8
    python3 evaluate-stn-bypass.py --images path/to/images --labels path/to/labels
    python3 evaluate-stn-bypass.py --precision fp32 --output results/stn-ablation
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, shape_inference

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "fp32": str(SCRIPT_DIR / "models/stn-fomo-480-onnx/stn-fomo-480.onnx"),
    "int8": str(SCRIPT_DIR / "models/stn-fomo-480-onnx/stn-fomo-480-int8.onnx"),
}
DEFAULT_IMAGES = str(SCRIPT_DIR / "datasets/yolo_pl_test/images")
DEFAULT_LABELS = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_OUTPUT  = str(SCRIPT_DIR / "benchmark_results/stn-bypass")
CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]


# ---------------------------------------------------------------------------
# Load the shared evaluation primitives from evaluate-fomo.py
# (preprocess, postprocess, match_for_map, update_confusion_matrix, metrics,
# save_confusion_png, save_metrics_csv, save_metrics_png, …)
# ---------------------------------------------------------------------------
def _load_eval_module():
    path = SCRIPT_DIR / "evaluate-fomo.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find evaluate-fomo.py at {path}. "
            "Run this script from the same directory."
        )
    spec = importlib.util.spec_from_file_location("evaluate_fomo", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# STN bypass: replace the GridSample grid with an identity constant
# ---------------------------------------------------------------------------
def _patch_bypass_stn(model: onnx.ModelProto) -> bool:
    """Modify the ONNX graph in-place so the STN applies an identity warp.

    Finds the unique GridSample node and replaces its grid input (the STN's
    learned sampling coordinates) with a constant identity grid, so every
    output pixel samples from the same input position — equivalent to θ=I.

    Works for both fp32 and int8 graphs because we overwrite the direct input
    to GridSample (always float32 in both variants).

    Returns True on success, False if no GridSample was found (non-STN model).
    """
    gs_node = next((n for n in model.graph.node if n.op_type == "GridSample"), None)
    if gs_node is None:
        return False

    # Detect spatial size from shape info; fall back to 30 (stn-fomo default)
    H = W = 30
    grid_input_name = gs_node.input[1]
    vmap = {vi.name: vi for vi in model.graph.value_info}
    if grid_input_name in vmap:
        shape = vmap[grid_input_name].type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            h_val = shape.dim[1].dim_value
            w_val = shape.dim[2].dim_value
            if h_val > 0 and w_val > 0:
                H, W = h_val, w_val

    # Identity grid (1, H, W, 2) — align_corners=False pixel-centre coords
    cols = (2.0 * np.arange(W, dtype=np.float32) + 1.0) / W - 1.0
    rows = (2.0 * np.arange(H, dtype=np.float32) + 1.0) / H - 1.0
    xs, ys = np.meshgrid(cols, rows)
    identity = np.stack([xs, ys], axis=-1)[np.newaxis]  # (1, H, W, 2)

    const_name = "__bypass_identity_grid__"
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[const_name],
        value=numpy_helper.from_array(identity, name=const_name),
    )
    gs_node.input[1] = const_name
    model.graph.node.insert(0, const_node)
    return True


def _make_session(model_path: str, bypass: bool, providers: list[str],
                  num_threads: int) -> tuple[ort.InferenceSession, int]:
    """Load and (optionally) patch the model, return (session, input_size)."""
    m = onnx.load(model_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass

    if bypass:
        ok = _patch_bypass_stn(m)
        if not ok:
            raise RuntimeError(
                f"Could not find a GridSample node in {model_path}. "
                "Is this actually an STN model?"
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

    sess = ort.InferenceSession(
        m.SerializeToString(), sess_options=opts, providers=configured
    )
    inp = sess.get_inputs()[0]
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 480
    return sess, input_size


# ---------------------------------------------------------------------------
# Evaluation loop  (mirrors evaluate-fomo.py::evaluate() but takes a session)
# ---------------------------------------------------------------------------
def run_evaluation(
    sess: ort.InferenceSession,
    input_size: int,
    images_dir: str,
    labels_dir: str,
    classes: list[str],
    ev,   # the imported evaluate-fomo module
    confmat_conf: float,
    confmat_dist: float,
    label: str = "",
) -> dict:
    """Run the full FOMO evaluation loop and return a result dict compatible
    with evaluate-fomo.py's reporting functions."""
    in_name = sess.get_inputs()[0].name
    nc      = len(classes)
    ndist   = len(ev.DISTANCE_THRESHOLDS)

    img_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in ev.IMG_EXTS
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    stats_tp   = [[] for _ in range(nc)]
    stats_conf = [[] for _ in range(nc)]
    n_gt       = np.zeros(nc, dtype=np.int64)
    cm         = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    inf_times  = []
    grid_w = grid_h = 0
    t_total = time.perf_counter()

    for done, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gts = [g for g in ev.load_gt(str(Path(labels_dir) / (img_path.stem + ".txt")))
               if 0 <= g[0] < nc]

        blob = ev.preprocess(img, input_size)
        t0   = time.perf_counter()
        out  = sess.run(None, {in_name: blob})[0]
        inf_times.append(time.perf_counter() - t0)

        if not grid_w:
            shp    = np.asarray(out).shape
            grid_h = int(shp[-2])
            grid_w = int(shp[-1])

        dets = ev.postprocess(out, nc, ev.CONF_THRESHOLD_INFER)
        dets.sort(key=lambda d: -d["score"])

        for gc, _ in gts:
            n_gt[gc] += 1

        tp_arr = ev.match_for_map(dets, gts, ndist)
        for di, d in enumerate(dets):
            stats_tp[d["cls"]].append(tp_arr[di])
            stats_conf[d["cls"]].append(d["score"])

        ev.update_confusion_matrix(cm, dets, gts, nc, confmat_conf, confmat_dist)

        if done % 50 == 0 or done == len(img_paths):
            print(f"  [{done:>4}/{len(img_paths)}] {label}")

    wall = time.perf_counter() - t_total

    ap     = np.zeros((nc, ndist), dtype=np.float64)
    p_best = np.zeros(nc)
    r_best = np.zeros(nc)
    f1_best = np.zeros(nc)
    thr_best = np.zeros(nc)
    primary_ti = int(np.argmin(np.abs(ev.DISTANCE_THRESHOLDS - ev.PRIMARY_DISTANCE)))

    for c in range(nc):
        if not stats_tp[c] or n_gt[c] == 0:
            continue
        tp   = np.array(stats_tp[c], dtype=bool)
        conf = np.array(stats_conf[c], dtype=np.float64)
        order = np.argsort(-conf)
        tp   = tp[order]
        conf = conf[order]
        fp   = ~tp
        tp_cum = tp.cumsum(0)
        fp_cum = fp.cumsum(0)
        recall    = tp_cum / max(n_gt[c], 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        for ti in range(ndist):
            ap[c, ti] = ev.compute_ap(recall[:, ti], precision[:, ti])

        rec_p = recall[:, primary_ti]
        pre_p = precision[:, primary_ti]
        f1_p  = 2 * pre_p * rec_p / np.maximum(pre_p + rec_p, 1e-16)
        idx   = int(np.argmax(f1_p))
        p_best[c]   = pre_p[idx]
        r_best[c]   = rec_p[idx]
        f1_best[c]  = f1_p[idx]
        thr_best[c] = conf[idx]

    return {
        "classes":           classes,
        "n_gt":              n_gt,
        "ap":                ap,
        "ap_primary":        float(ap[:, primary_ti].mean()),
        "ap_mean":           float(ap.mean()),
        "distance_thresholds": ev.DISTANCE_THRESHOLDS,
        "primary_ti":        primary_ti,
        "primary_distance":  float(ev.DISTANCE_THRESHOLDS[primary_ti]),
        "precision":         p_best,
        "recall":            r_best,
        "f1":                f1_best,
        "conf_thr":          thr_best,
        "confusion_matrix":  cm,
        "inference_ms_mean": 1000.0 * float(np.mean(inf_times)) if inf_times else 0.0,
        "inference_ms_p95":  1000.0 * float(np.percentile(inf_times, 95)) if inf_times else 0.0,
        "wall_seconds":      wall,
        "input_size":        input_size,
        "grid_size":         (grid_w, grid_h),
        "num_images":        len(img_paths),
        "confmat_conf":      confmat_conf,
        "confmat_dist":      confmat_dist,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_COL = 24  # column width for class names

def _fmt(v, w=9, d=4):
    return f"{v:{w}.{d}f}"

def _delta_str(a, b, invert=False):
    """Format a - b with colour-coded + / - sign."""
    diff = a - b
    if invert:
        diff = -diff
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:+.4f}"


def print_comparison(ra, rb, name_a="STN-active", name_b="STN-bypass"):
    classes = ra["classes"]
    nc = len(classes)
    pri = ra["primary_ti"]
    dist = ra["primary_distance"]

    banner = f"  {'class':<{_COL}}  {'|':1}  {'':>8}  {name_a:>8}  {'|':1}  {'':>8}  {name_b:>8}  {'|':1}  {'Δ (active−bypass)':>18}"
    sub    = f"  {'':_<{_COL}}  {'|':1}  {'P':>8}  {'R':>8}  {'F1':>8}  {'AP@d':>8}  {'|':1}  {'P':>8}  {'R':>8}  {'F1':>8}  {'AP@d':>8}  {'|':1}  {'ΔP':>8}  {'ΔR':>8}  {'ΔF1':>8}  {'ΔAP':>8}"

    print(f"\n{'='*len(sub)}")
    print(f"  STN-FOMO  ·  {name_a} vs {name_b}  ·  distance threshold = {dist:.3f} norm")
    print(f"{'='*len(sub)}")
    print(sub)
    print(f"  {'-'*(_COL+2)}{'|':1}{'-'*19}{'|':1}{'-'*19}{'|':1}{'-'*35}")

    for c, cname in enumerate(classes):
        pa, ra_c, f1a, apa = (ra["precision"][c], ra["recall"][c],
                               ra["f1"][c], ra["ap"][c, pri])
        pb, rb_c, f1b, apb = (rb["precision"][c], rb["recall"][c],
                               rb["f1"][c], rb["ap"][c, pri])
        print(
            f"  {cname:<{_COL}}  |"
            f"  {pa:8.4f}  {ra_c:8.4f}  {f1a:8.4f}  {apa:8.4f}  |"
            f"  {pb:8.4f}  {rb_c:8.4f}  {f1b:8.4f}  {apb:8.4f}  |"
            f"  {pa-pb:+8.4f}  {ra_c-rb_c:+8.4f}  {f1a-f1b:+8.4f}  {apa-apb:+8.4f}"
        )

    # mean row
    def _mean(r, attr):
        return r[attr].mean()
    print(f"  {'-'*(_COL+2)}{'|':1}{'-'*19}{'|':1}{'-'*19}{'|':1}{'-'*35}")
    print(
        f"  {'MEAN':<{_COL}}  |"
        f"  {_mean(ra,'precision'):8.4f}  {_mean(ra,'recall'):8.4f}"
        f"  {_mean(ra,'f1'):8.4f}  {ra['ap_primary']:8.4f}  |"
        f"  {_mean(rb,'precision'):8.4f}  {_mean(rb,'recall'):8.4f}"
        f"  {_mean(rb,'f1'):8.4f}  {rb['ap_primary']:8.4f}  |"
        f"  {_mean(ra,'precision')-_mean(rb,'precision'):+8.4f}"
        f"  {_mean(ra,'recall')-_mean(rb,'recall'):+8.4f}"
        f"  {_mean(ra,'f1')-_mean(rb,'f1'):+8.4f}"
        f"  {ra['ap_primary']-rb['ap_primary']:+8.4f}"
    )
    print(f"{'='*len(sub)}")


def print_insights(ra, rb, classes):
    """Print a plain-English summary of what the STN contributes."""
    nc = len(classes)
    pri = ra["primary_ti"]

    f1_delta  = ra["f1"]  - rb["f1"]
    ap_delta  = ra["ap"][:, pri] - rb["ap"][:, pri]
    rec_delta = ra["recall"] - rb["recall"]
    pre_delta = ra["precision"] - rb["precision"]

    best_c    = int(np.argmax(f1_delta))
    worst_c   = int(np.argmin(f1_delta))

    # Confusion-matrix derived TP / FP / FN
    cm_a, cm_b = ra["confusion_matrix"], rb["confusion_matrix"]
    tp_a = np.diag(cm_a[:nc, :nc]).sum()
    fn_a = cm_a[:nc, nc].sum()
    fp_a = cm_a[nc, :nc].sum()
    tp_b = np.diag(cm_b[:nc, :nc]).sum()
    fn_b = cm_b[:nc, nc].sum()
    fp_b = cm_b[nc, :nc].sum()

    # Per-class TP / FN
    tp_class_a = np.diag(cm_a[:nc, :nc])
    tp_class_b = np.diag(cm_b[:nc, :nc])
    fn_class_a = cm_a[:nc, nc]
    fn_class_b = cm_b[:nc, nc]

    static_ids = [i for i in range(nc) if i != 0]  # all except railroad-crossing

    print("\n" + "─" * 72)
    print("  INSIGHTS")
    print("─" * 72)

    # Overall
    print(f"\n  Overall (mean across all {nc} classes):")
    print(f"    F1  : {ra['f1'].mean():.4f}  (active)  vs  {rb['f1'].mean():.4f}  (bypass)"
          f"  →  Δ = {ra['f1'].mean()-rb['f1'].mean():+.4f}")
    print(f"    AP  : {ra['ap_primary']:.4f}  (active)  vs  {rb['ap_primary']:.4f}  (bypass)"
          f"  →  Δ = {ra['ap_primary']-rb['ap_primary']:+.4f}")
    print(f"    TP  : {tp_a}  (active)  vs  {tp_b}  (bypass)"
          f"  →  STN recovers {tp_a - tp_b:+d} extra TPs")
    print(f"    FN  : {fn_a}  (active)  vs  {fn_b}  (bypass)"
          f"  →  STN suppresses {fn_b - fn_a:+d} missed detections")
    print(f"    FP  : {fp_a}  (active)  vs  {fp_b}  (bypass)")

    # Class that benefits most
    print(f"\n  Class that benefits MOST from the STN:")
    print(f"    {classes[best_c]}"
          f"  ΔF1={f1_delta[best_c]:+.4f}"
          f"  ΔR={rec_delta[best_c]:+.4f}"
          f"  ΔP={pre_delta[best_c]:+.4f}"
          f"  ΔAP={ap_delta[best_c]:+.4f}")
    print(f"    TPs: {tp_class_a[best_c]}  (active)  vs  {tp_class_b[best_c]}  (bypass)"
          f"  |  FNs: {fn_class_a[best_c]} vs {fn_class_b[best_c]}")

    # Static classes
    static_delta = f1_delta[static_ids]
    print(f"\n  Static classes (not expected to rotate):")
    for i in static_ids:
        sign_str = "improved" if f1_delta[i] > 0.01 else ("hurt" if f1_delta[i] < -0.01 else "unchanged")
        print(f"    {classes[i]:<22}  ΔF1={f1_delta[i]:+.4f}  ({sign_str})")

    # Diagnosis
    print(f"\n  Diagnosis:")
    if f1_delta[0] > 0.05:
        print(f"    ✓ The STN provides a meaningful boost to railroad-crossing detection"
              f" (+{f1_delta[0]:.3f} F1), confirming the rotation-normalisation is effective.")
    elif f1_delta[0] > 0:
        print(f"    ○ The STN provides a modest improvement to railroad-crossing detection"
              f" (+{f1_delta[0]:.3f} F1).")
    else:
        print(f"    ✗ The STN does NOT improve railroad-crossing detection on this test set"
              f" ({f1_delta[0]:.3f} F1). Consider whether the test set covers enough rotation variance.")

    max_static_drop = min(static_delta)
    if max_static_drop < -0.05:
        worst_static = static_ids[int(np.argmin(static_delta))]
        print(f"    ✗ Static class '{classes[worst_static]}' is hurt by the STN"
              f" ({f1_delta[worst_static]:.3f} F1) — check for warp over-correction.")
    else:
        print(f"    ✓ Static class F1 changes by at most {abs(max_static_drop):.3f}"
              f" — the STN does not meaningfully hurt orientation-invariant classes.")

    recall_gain  = rec_delta[0]
    precis_change = pre_delta[0]
    if abs(recall_gain) > abs(precis_change):
        print(f"    ✓ Improvement is recall-dominant (ΔR={recall_gain:+.4f}"
              f" vs ΔP={precis_change:+.4f}): the STN recovers detections that"
              f" the bypass misses, not just reducing false positives.")
    else:
        print(f"    ○ Improvement is precision-dominant (ΔP={precis_change:+.4f}"
              f" vs ΔR={recall_gain:+.4f}): the STN mainly reduces false positives.")

    print("─" * 72)


def save_comparison_png(path, ra, rb, classes, name_a, name_b):
    """Save a side-by-side bar chart of F1 per class for both modes."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable — skipping comparison.png)")
        return

    nc = len(classes)
    x  = np.arange(nc)
    w  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1 bars
    ax = axes[0]
    bars_a = ax.bar(x - w/2, ra["f1"], w, label=name_a, color="#2563EB", alpha=0.85)
    bars_b = ax.bar(x + w/2, rb["f1"], w, label=name_b, color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1")
    ax.set_title("F1 per class — STN active vs bypass", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    # Right: ΔF1 bars
    ax2 = axes[1]
    delta = ra["f1"] - rb["f1"]
    colours = ["#16A34A" if d >= 0 else "#DC2626" for d in delta]
    ax2.bar(x, delta, color=colours, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("ΔF1  (active − bypass)")
    ax2.set_title("F1 gain from the STN warp per class", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    for xi, d in zip(x, delta):
        ax2.text(xi, d + (0.003 if d >= 0 else -0.012),
                 f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                 fontsize=8, fontweight="bold")

    fig.suptitle(
        f"STN-FOMO ablation  ·  {len(ra['n_gt'])} classes  ·  {ra['num_images']} test images",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] comparison chart → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32",
                    help="Which STN-FOMO variant to evaluate (default: fp32).")
    ap.add_argument("--images",  default=DEFAULT_IMAGES,
                    help="Directory of test images.")
    ap.add_argument("--labels",  default=DEFAULT_LABELS,
                    help="Directory of YOLO label files (.txt, OBB or centroid).")
    ap.add_argument("--output",  default=DEFAULT_OUTPUT,
                    help="Output directory for CSV / PNG artifacts.")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                    help="ONNX Runtime execution providers.")
    ap.add_argument("--threads",   type=int, default=4,
                    help="Intra-op threads for ONNX Runtime.")
    ap.add_argument("--confmat-conf", type=float, default=0.25,
                    help="Confidence threshold for confusion-matrix matching.")
    ap.add_argument("--confmat-dist", type=float, default=0.10,
                    help="Normalised distance threshold for confusion-matrix matching.")
    return ap.parse_args()


def main():
    args = parse_args()

    model_path = MODEL_PATHS.get(args.precision)
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    os.makedirs(args.output, exist_ok=True)

    ev = _load_eval_module()

    # -----------------------------------------------------------------------
    # Build sessions
    # -----------------------------------------------------------------------
    print(f"\n[1/4] Loading STN-FOMO {args.precision.upper()} — normal session …")
    sess_active, input_size = _make_session(
        model_path, bypass=False,
        providers=args.providers, num_threads=args.threads,
    )

    print(f"[2/4] Loading STN-FOMO {args.precision.upper()} — bypass session (identity grid) …")
    sess_bypass, _ = _make_session(
        model_path, bypass=True,
        providers=args.providers, num_threads=args.threads,
    )

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    print(f"\n[3/4] Evaluating STN-active on {args.images} …")
    result_active = run_evaluation(
        sess_active, input_size, args.images, args.labels, CLASSES, ev,
        confmat_conf=args.confmat_conf,
        confmat_dist=args.confmat_dist,
        label="STN-active",
    )

    print(f"\n[4/4] Evaluating STN-bypass on {args.images} …")
    result_bypass = run_evaluation(
        sess_bypass, input_size, args.images, args.labels, CLASSES, ev,
        confmat_conf=args.confmat_conf,
        confmat_dist=args.confmat_dist,
        label="STN-bypass",
    )

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    name_a = f"STN-FOMO-{args.precision.upper()} (active)"
    name_b = f"STN-FOMO-{args.precision.upper()} (bypass)"

    print_comparison(result_active, result_bypass, name_a=name_a, name_b=name_b)
    ev.print_confusion_matrix(name_a, result_active["confusion_matrix"],
                               CLASSES, args.confmat_conf, args.confmat_dist)
    ev.print_confusion_matrix(name_b, result_bypass["confusion_matrix"],
                               CLASSES, args.confmat_conf, args.confmat_dist)
    print_insights(result_active, result_bypass, CLASSES)

    # -----------------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------------
    slug = f"stn-fomo-{args.precision}"
    ev.save_metrics_csv(
        os.path.join(args.output, f"metrics_active_{slug}.csv"), name_a, result_active)
    ev.save_metrics_csv(
        os.path.join(args.output, f"metrics_bypass_{slug}.csv"), name_b, result_bypass)
    ev.save_metrics_png(
        os.path.join(args.output, f"metrics_active_{slug}.png"), name_a, result_active)
    ev.save_metrics_png(
        os.path.join(args.output, f"metrics_bypass_{slug}.png"), name_b, result_bypass)
    ev.save_confusion_png(
        os.path.join(args.output, f"confusion_active_{slug}.png"),
        result_active["confusion_matrix"], CLASSES, name_a)
    ev.save_confusion_png(
        os.path.join(args.output, f"confusion_bypass_{slug}.png"),
        result_bypass["confusion_matrix"], CLASSES, name_b)
    save_comparison_png(
        os.path.join(args.output, f"comparison_{slug}.png"),
        result_active, result_bypass, CLASSES, name_a, name_b)

    print(f"\nArtifacts written to: {args.output}")


if __name__ == "__main__":
    main()
