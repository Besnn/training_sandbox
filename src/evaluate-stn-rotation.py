#!/usr/bin/env python3
"""STN-FOMO rotation-robustness sweep: F1 / AP vs rotation angle.

For each rotation angle θ in a configurable sweep every test image is rotated
by θ degrees counter-clockwise and the same model is evaluated in two modes:

  STN-active  — learned spatial transformer applied (the real model)
  STN-bypass  — identity grid, backbone output goes straight to the head

Plotting F1 vs θ shows whether the STN keeps detection stable across angles
(flat active curve) while the bypass degrades (falling bypass curve).  That
divergence is the primary empirical evidence that the STN's rotation
normalisation is effective.

Ground-truth centroids are rotated by the same angle around the image centre
so the metric stays meaningful.  Annotations that rotate outside the unit
square are filtered — the object is no longer in frame.

Usage
-----
    python3 evaluate-stn-rotation.py
    python3 evaluate-stn-rotation.py --angles 0 30 60 90 120 150 180
    python3 evaluate-stn-rotation.py --precision int8 --angles $(seq 0 15 180)
    python3 evaluate-stn-rotation.py --images path/to/images --labels path/to/labels
"""
from __future__ import annotations

import argparse
import csv
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
# Paths / defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "fp32": str(SCRIPT_DIR / "models/stn-fomo-480-onnx/stn-fomo-480.onnx"),
    "int8": str(SCRIPT_DIR / "models/stn-fomo-480-onnx/stn-fomo-480-int8.onnx"),
}
DEFAULT_IMAGES  = str(SCRIPT_DIR / "datasets/yolo_pl_test/images")
DEFAULT_LABELS  = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_OUTPUT  = str(SCRIPT_DIR / "benchmark_results/stn-rotation")
CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# Default sweep: symmetric ±90° in 15° steps plus 0°
# Use 0–180 to show the full half-period of the crossing's bilateral symmetry.
DEFAULT_ANGLES = list(range(0, 181, 15))


# ---------------------------------------------------------------------------
# Import shared evaluation primitives from evaluate-fomo.py
# ---------------------------------------------------------------------------
def _load_eval_module():
    path = SCRIPT_DIR / "evaluate-fomo.py"
    if not path.is_file():
        raise FileNotFoundError(f"evaluate-fomo.py not found at {path}")
    spec = importlib.util.spec_from_file_location("evaluate_fomo", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# STN bypass patch  (identical to app.py / evaluate-stn-bypass.py)
# ---------------------------------------------------------------------------
def _patch_bypass_stn(model: onnx.ModelProto) -> bool:
    gs_node = next((n for n in model.graph.node if n.op_type == "GridSample"), None)
    if gs_node is None:
        return False

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

    cols = (2.0 * np.arange(W, dtype=np.float32) + 1.0) / W - 1.0
    rows = (2.0 * np.arange(H, dtype=np.float32) + 1.0) / H - 1.0
    xs, ys = np.meshgrid(cols, rows)
    identity = np.stack([xs, ys], axis=-1)[np.newaxis]

    const_name = "__bypass_identity_grid__"
    const_node = helper.make_node(
        "Constant", inputs=[], outputs=[const_name],
        value=numpy_helper.from_array(identity, name=const_name),
    )
    gs_node.input[1] = const_name
    model.graph.node.insert(0, const_node)
    return True


def _make_session(model_path: str, bypass: bool, providers: list[str],
                  num_threads: int) -> tuple[ort.InferenceSession, int]:
    m = onnx.load(model_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass
    if bypass:
        _patch_bypass_stn(m)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    configured = [
        (p, {"intra_op_num_threads": str(num_threads)})
        if p == "XnnpackExecutionProvider" else p
        for p in providers
    ]
    sess = ort.InferenceSession(
        m.SerializeToString(), sess_options=opts, providers=configured
    )
    inp = sess.get_inputs()[0]
    try:
        sz = int(inp.shape[2])
    except (TypeError, ValueError):
        sz = 480
    return sess, sz


# ---------------------------------------------------------------------------
# Image and label rotation helpers
# ---------------------------------------------------------------------------
def rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate img by angle_deg degrees CCW around its centre.

    The canvas stays the same size; pixels that rotate outside get filled
    with black (constant 0).  The same transform is applied to GT centroids
    by rotate_gt() below so evaluation remains coherent.
    """
    if angle_deg == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def rotate_gt(gts: list[tuple[int, tuple[float, float]]],
              angle_deg: float) -> list[tuple[int, tuple[float, float]]]:
    """Rotate GT centroids by angle_deg CCW around (0.5, 0.5).

    Annotations whose centroid rotates outside [0, 1]² are dropped — the
    corresponding object is no longer visible in the rotated image.

    This must use the IDENTICAL transform as rotate_image() above.
    A normalized centroid (cx, cy) in [0,1]² maps to pixel space as
    (cx * W, cy * H); rotating by angle_deg CCW around (W/2, H/2) and
    mapping back to normalized:

        dx = cx - 0.5,  dy = cy - 0.5
        cx' = 0.5 + dx·cos θ - dy·sin θ
        cy' = 0.5 + dx·sin θ + dy·cos θ

    The formula is correct because the image is square (W = H = 480).
    """
    if angle_deg == 0:
        return gts

    rad   = np.radians(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)

    out = []
    for cls_id, (cx, cy) in gts:
        dx, dy = cx - 0.5, cy - 0.5
        cx2 = 0.5 + dx * cos_a - dy * sin_a
        cy2 = 0.5 + dx * sin_a + dy * cos_a
        if 0.0 <= cx2 <= 1.0 and 0.0 <= cy2 <= 1.0:
            out.append((cls_id, (cx2, cy2)))
    return out


# ---------------------------------------------------------------------------
# Evaluation loop at a single angle
# ---------------------------------------------------------------------------
def evaluate_at_angle(
    sess: ort.InferenceSession,
    input_size: int,
    angle_deg: float,
    images_dir: str,
    labels_dir: str,
    classes: list[str],
    ev,
    confmat_conf: float = 0.25,
    confmat_dist: float = 0.10,
) -> dict:
    """Run the FOMO evaluation loop with every image rotated by angle_deg."""
    in_name = sess.get_inputs()[0].name
    nc      = len(classes)
    ndist   = len(ev.DISTANCE_THRESHOLDS)

    img_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in ev.IMG_EXTS
    )

    stats_tp   = [[] for _ in range(nc)]
    stats_conf = [[] for _ in range(nc)]
    n_gt       = np.zeros(nc, dtype=np.int64)
    cm         = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    inf_times  = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Load and rotate ground truth
        raw_gts = [g for g in ev.load_gt(
                       str(Path(labels_dir) / (img_path.stem + ".txt")))
                   if 0 <= g[0] < nc]
        gts = rotate_gt(raw_gts, angle_deg)

        # Rotate image and run inference
        img_rot = rotate_image(img, angle_deg)
        blob    = ev.preprocess(img_rot, input_size)
        t0      = time.perf_counter()
        out     = sess.run(None, {in_name: blob})[0]
        inf_times.append(time.perf_counter() - t0)

        dets = ev.postprocess(out, nc, ev.CONF_THRESHOLD_INFER)
        dets.sort(key=lambda d: -d["score"])

        for gc, _ in gts:
            n_gt[gc] += 1

        tp_arr = ev.match_for_map(dets, gts, ndist)
        for di, d in enumerate(dets):
            stats_tp[d["cls"]].append(tp_arr[di])
            stats_conf[d["cls"]].append(d["score"])

        ev.update_confusion_matrix(cm, dets, gts, nc, confmat_conf, confmat_dist)

    primary_ti = int(np.argmin(np.abs(ev.DISTANCE_THRESHOLDS - ev.PRIMARY_DISTANCE)))
    ap     = np.zeros((nc, ndist), dtype=np.float64)
    p_best = np.zeros(nc)
    r_best = np.zeros(nc)
    f1_best = np.zeros(nc)

    for c in range(nc):
        if not stats_tp[c] or n_gt[c] == 0:
            continue
        tp   = np.array(stats_tp[c], dtype=bool)
        conf = np.array(stats_conf[c], dtype=np.float64)
        order = np.argsort(-conf)
        tp   = tp[order]
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
        p_best[c]  = pre_p[idx]
        r_best[c]  = rec_p[idx]
        f1_best[c] = f1_p[idx]

    return {
        "angle":       angle_deg,
        "n_gt":        n_gt,
        "precision":   p_best,
        "recall":      r_best,
        "f1":          f1_best,
        "ap":          ap,
        "ap_primary":  float(ap[:, primary_ti].mean()),
        "ms_mean":     1000.0 * float(np.mean(inf_times)) if inf_times else 0.0,
        "n_imgs":      len(img_paths),
        "n_gt_visible": int(n_gt.sum()),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_sweep_table(sweep_active, sweep_bypass, classes, metric="f1"):
    """Print F1 (or other metric) per class per angle for both modes."""
    nc     = len(classes)
    angles = [r["angle"] for r in sweep_active]

    col_w = 8
    head  = f"  {'angle':>6}  " + "  ".join(
        f"{'[A]' + c[:7]:>{col_w}}  {'[B]' + c[:7]:>{col_w}}  {'Δ':>{col_w}}"
        for c in classes
    ) + f"  {'[A]mean':>{col_w}}  {'[B]mean':>{col_w}}  {'Δmean':>{col_w}}"

    print(f"\n{'='*len(head)}")
    print(f"  {metric.upper()} per class — [A]=STN-active  [B]=STN-bypass  Δ=A−B")
    print(f"{'='*len(head)}")
    print(head)
    print(f"  {'-'*6}  " + "  ".join([f"{'-'*(3*col_w+4)}"] * nc) + f"  {'-'*(3*col_w+4)}")

    for ra, rb in zip(sweep_active, sweep_bypass):
        vals_a = ra[metric]
        vals_b = rb[metric]
        row = f"  {ra['angle']:>6}°  "
        for c in range(nc):
            a, b = vals_a[c], vals_b[c]
            row += f"  {a:{col_w}.4f}  {b:{col_w}.4f}  {a-b:+{col_w}.4f}"
        ma, mb = vals_a.mean(), vals_b.mean()
        row += f"  {ma:{col_w}.4f}  {mb:{col_w}.4f}  {ma-mb:+{col_w}.4f}"
        print(row)

    print(f"{'='*len(head)}")


def print_insights(sweep_active, sweep_bypass, classes):
    """Print a concise narrative of what the curves show."""
    nc     = len(classes)
    angles = np.array([r["angle"] for r in sweep_active])
    n      = len(angles)

    # F1 arrays: shape (n_angles, n_classes)
    f1_a = np.stack([r["f1"] for r in sweep_active])
    f1_b = np.stack([r["f1"] for r in sweep_bypass])

    # Baseline (0° or nearest)
    base_idx = int(np.argmin(np.abs(angles - 0)))

    f1_drop_a = f1_a[base_idx] - f1_a.min(axis=0)   # max drop for active
    f1_drop_b = f1_b[base_idx] - f1_b.min(axis=0)   # max drop for bypass

    # Angle of largest drop for railroad-crossing
    worst_angle_a = angles[int(np.argmin(f1_a[:, 0]))]
    worst_angle_b = angles[int(np.argmin(f1_b[:, 0]))]

    # Mean F1 across the whole sweep
    mean_f1_a = f1_a[:, 0].mean()
    mean_f1_b = f1_b[:, 0].mean()

    print("\n" + "─" * 72)
    print("  ROTATION ROBUSTNESS — INSIGHTS")
    print("─" * 72)

    print(f"\n  railroad-crossing  (the rotating class)")
    print(f"    STN-active  : baseline F1={f1_a[base_idx,0]:.4f}  "
          f"worst F1={f1_a[:,0].min():.4f} @ {worst_angle_a}°  "
          f"max drop={f1_drop_a[0]:.4f}  mean={mean_f1_a:.4f}")
    print(f"    STN-bypass  : baseline F1={f1_b[base_idx,0]:.4f}  "
          f"worst F1={f1_b[:,0].min():.4f} @ {worst_angle_b}°  "
          f"max drop={f1_drop_b[0]:.4f}  mean={mean_f1_b:.4f}")
    print(f"    STN advantage (mean F1 across sweep): "
          f"{mean_f1_a - mean_f1_b:+.4f}")

    print(f"\n  Static classes  (should be unaffected by the STN)")
    for c in range(1, nc):
        print(f"    {classes[c]:<22}  "
              f"active drop={f1_drop_a[c]:.4f}  "
              f"bypass drop={f1_drop_b[c]:.4f}  "
              f"mean Δ={(f1_a[:,c]-f1_b[:,c]).mean():+.4f}")

    print(f"\n  Diagnosis:")
    advantage = mean_f1_a - mean_f1_b
    if advantage > 0.05:
        verdict = (f"✓  Strong:  STN-active maintains +{advantage:.3f} higher mean F1 "
                   f"for railroad-crossing across the {angles[0]}°–{angles[-1]}° sweep. "
                   f"The rotation normalisation is effective.")
    elif advantage > 0.01:
        verdict = (f"○  Moderate:  STN-active is +{advantage:.3f} higher on average, "
                   f"confirming partial rotation benefit.")
    else:
        verdict = (f"✗  Weak:  STN-active is only +{advantage:.3f} better on average. "
                   f"The test-set rotation variance may be insufficient, or the STN "
                   f"is not correcting for the angles tested.")
    print(f"    {verdict}")

    drop_ratio = f1_drop_b[0] / max(f1_drop_a[0], 1e-6)
    if drop_ratio > 2:
        print(f"    ✓  The bypass degrades {drop_ratio:.1f}× more than the active model "
              f"({f1_drop_b[0]:.3f} vs {f1_drop_a[0]:.3f} F1 drop), "
              f"confirming the STN provides rotation resilience.")
    elif f1_drop_a[0] > 0.05:
        print(f"    ○  Both modes degrade at high angles, but the active model "
              f"degrades less ({f1_drop_a[0]:.3f} vs {f1_drop_b[0]:.3f}).")
    else:
        print(f"    ○  Neither mode degrades much — the test set may not contain "
              f"enough rotation variance to stress-test the STN.")

    print("─" * 72)


def save_sweep_csv(path: str, sweep_active, sweep_bypass, classes):
    """One row per (angle, mode, class) — easy to load into a spreadsheet."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle_deg", "mode", "class", "precision", "recall",
                    "f1", "ap", "mean_ap", "n_gt_visible"])
        for mode_name, sweep in [("active", sweep_active), ("bypass", sweep_bypass)]:
            for r in sweep:
                for c, cname in enumerate(classes):
                    w.writerow([
                        r["angle"], mode_name, cname,
                        f"{r['precision'][c]:.6f}",
                        f"{r['recall'][c]:.6f}",
                        f"{r['f1'][c]:.6f}",
                        f"{r['ap'][c, 0]:.6f}",
                        f"{r['ap_primary']:.6f}",
                        r["n_gt_visible"],
                    ])
    print(f"  [OK] sweep CSV → {path}")


def save_degradation_plot(path: str, sweep_active, sweep_bypass,
                          classes, precision, metric="f1"):
    """Main thesis figure: F1 vs rotation angle, active vs bypass per class."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        print("  (matplotlib unavailable — skipping degradation plot)")
        return

    angles  = np.array([r["angle"] for r in sweep_active])
    nc      = len(classes)
    f1_a    = np.stack([r[metric] for r in sweep_active])   # (n_angles, nc)
    f1_b    = np.stack([r[metric] for r in sweep_bypass])

    # Colour palette — distinct per class
    colours = ["#2563EB", "#16A34A", "#D97706", "#9333EA"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ---- Left: per-class F1 curves ----------------------------------------
    ax = axes[0]
    for c in range(nc):
        col = colours[c % len(colours)]
        ax.plot(angles, f1_a[:, c], color=col,  linewidth=2.2,
                marker="o", markersize=5, label=f"{classes[c]}  [STN-active]")
        ax.plot(angles, f1_b[:, c], color=col,  linewidth=1.5,
                linestyle="--", marker="s", markersize=4,
                label=f"{classes[c]}  [STN-bypass]", alpha=0.7)

    ax.set_xlabel("Rotation angle  (degrees CCW)", fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(angles[0] - 3, angles[-1] + 3)
    ax.set_xticks(angles)
    ax.set_title("Per-class F1 vs rotation angle", fontweight="bold")
    ax.grid(alpha=0.3)
    # Custom legend: solid=active, dashed=bypass
    legend_elems = (
        [Line2D([0],[0], color=colours[c], lw=2, label=classes[c]) for c in range(nc)] +
        [Line2D([0],[0], color="k", lw=2, label="solid = STN-active"),
         Line2D([0],[0], color="k", lw=1.5, linestyle="--", label="dashed = STN-bypass")]
    )
    ax.legend(handles=legend_elems, fontsize=8, loc="lower left")

    # ---- Right: railroad-crossing only — zoomed + ΔF1 fill ----------------
    ax2 = axes[1]
    c = 0  # railroad-crossing
    ax2.plot(angles, f1_a[:, c], color=colours[c], linewidth=2.5,
             marker="o", markersize=6, label="STN-active", zorder=3)
    ax2.plot(angles, f1_b[:, c], color=colours[c], linewidth=1.8,
             linestyle="--", marker="s", markersize=5,
             label="STN-bypass", alpha=0.8, zorder=3)
    ax2.fill_between(angles, f1_b[:, c], f1_a[:, c],
                     alpha=0.18, color=colours[c],
                     label="STN advantage")

    ax2.set_xlabel("Rotation angle  (degrees CCW)", fontsize=11)
    ax2.set_ylabel("F1", fontsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.set_xlim(angles[0] - 3, angles[-1] + 3)
    ax2.set_xticks(angles)
    ax2.set_title(f"{classes[0]} — F1 vs rotation angle", fontweight="bold")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)

    # annotation: mean advantage
    adv = float((f1_a[:, c] - f1_b[:, c]).mean())
    ax2.text(
        angles[-1] * 0.55,
        0.06,
        f"Mean STN advantage: {adv:+.4f} F1",
        fontsize=9, color=colours[c],
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        f"STN-FOMO rotation robustness  ·  {precision.upper()}  ·  "
        f"{sweep_active[0]['n_imgs']} images  ·  "
        f"{int(angles[0])}°–{int(angles[-1])}° sweep",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] degradation plot → {path}")


def save_heatmap_plot(path: str, sweep_active, sweep_bypass, classes):
    """Heatmap: rows = class, cols = angle, cells = ΔF1 (active − bypass)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable — skipping heatmap)")
        return

    angles = np.array([r["angle"] for r in sweep_active])
    nc     = len(classes)
    f1_a   = np.stack([r["f1"] for r in sweep_active])
    f1_b   = np.stack([r["f1"] for r in sweep_bypass])
    delta  = f1_a - f1_b   # shape (n_angles, nc)

    fig, ax = plt.subplots(figsize=(max(8, len(angles) * 0.8), nc * 1.2 + 1.5))
    im = ax.imshow(delta.T, cmap="RdYlGn", vmin=-0.2, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(angles)))
    ax.set_xticklabels([f"{a}°" for a in angles])
    ax.set_yticks(range(nc))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Rotation angle")
    ax.set_title("ΔF1 (STN-active − STN-bypass)  ·  green = STN helps  ·  red = STN hurts",
                 fontweight="bold")
    for ci in range(nc):
        for ai in range(len(angles)):
            v = delta[ai, ci]
            ax.text(ai, ci, f"{v:+.3f}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.1 else "white")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ΔF1")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] delta heatmap → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--angles", type=float, nargs="+", default=DEFAULT_ANGLES,
                    metavar="DEG",
                    help="Rotation angles in degrees CCW (default: 0 15 30 … 180).")
    ap.add_argument("--images",  default=DEFAULT_IMAGES)
    ap.add_argument("--labels",  default=DEFAULT_LABELS)
    ap.add_argument("--output",  default=DEFAULT_OUTPUT)
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    ap.add_argument("--threads",   type=int, default=4)
    ap.add_argument("--confmat-conf", type=float, default=0.25)
    ap.add_argument("--confmat-dist", type=float, default=0.10)
    return ap.parse_args()


def main():
    args   = parse_args()
    angles = sorted(set(args.angles))

    model_path = MODEL_PATHS.get(args.precision)
    if not model_path or not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.isdir(args.images):
        raise SystemExit(f"Images dir not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels dir not found: {args.labels}")

    os.makedirs(args.output, exist_ok=True)

    ev = _load_eval_module()

    print(f"\nLoading STN-FOMO {args.precision.upper()} sessions …")
    sess_active, input_size = _make_session(
        model_path, bypass=False, providers=args.providers, num_threads=args.threads)
    sess_bypass, _ = _make_session(
        model_path, bypass=True,  providers=args.providers, num_threads=args.threads)
    print(f"Input size: {input_size}px  |  {len(angles)} angles: {angles}")

    sweep_active = []
    sweep_bypass = []
    total = len(angles) * 2
    done  = 0

    for angle in angles:
        done += 1
        print(f"\n[{done:>2}/{total}] angle={angle:+.0f}°  STN-active …")
        sweep_active.append(evaluate_at_angle(
            sess_active, input_size, angle,
            args.images, args.labels, CLASSES, ev,
            confmat_conf=args.confmat_conf,
            confmat_dist=args.confmat_dist,
        ))
        done += 1
        print(f"[{done:>2}/{total}] angle={angle:+.0f}°  STN-bypass …")
        sweep_bypass.append(evaluate_at_angle(
            sess_bypass, input_size, angle,
            args.images, args.labels, CLASSES, ev,
            confmat_conf=args.confmat_conf,
            confmat_dist=args.confmat_dist,
        ))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    print_sweep_table(sweep_active, sweep_bypass, CLASSES, metric="f1")
    print_insights(sweep_active, sweep_bypass, CLASSES)

    # -----------------------------------------------------------------------
    # Artifacts
    # -----------------------------------------------------------------------
    slug = f"stn-fomo-{args.precision}"
    save_sweep_csv(
        os.path.join(args.output, f"sweep_{slug}.csv"),
        sweep_active, sweep_bypass, CLASSES,
    )
    save_degradation_plot(
        os.path.join(args.output, f"degradation_{slug}.png"),
        sweep_active, sweep_bypass, CLASSES, args.precision, metric="f1",
    )
    save_heatmap_plot(
        os.path.join(args.output, f"delta_heatmap_{slug}.png"),
        sweep_active, sweep_bypass, CLASSES,
    )
    print(f"\nArtifacts written to: {args.output}")


if __name__ == "__main__":
    main()
