#!/usr/bin/env python3
"""STN-FOMO ablation stratified by railroad-crossing gate tilt angle.

Each test image is assigned to a tilt bin based on the OBB angle of its
railroad-crossing annotation.  The STN-active vs STN-bypass comparison is
then run once per bin so we can see whether the STN's benefit is concentrated
in a specific part of the gate's arc.

Tilt is measured as the angle of the gate's long axis from horizontal,
normalised to [0°, 90°]:

    0°  = fully horizontal  (gate closed)
    90° = fully vertical    (gate open)

Bins (configurable via --bins):
    0–10°    near-horizontal
    10–30°   shallow tilt
    30–60°   steep tilt
    60–80°   near-vertical
    80–90°   fully open

The prediction: ΔF1 (active − bypass) should peak in the 30–60° or 60–80°
bins where the gate's appearance is most geometrically ambiguous and the STN's
rotation correction matters most.

Outputs
-------
  stdout                                — per-bin table for railroad-crossing
                                          and a summary across all bins
  <output>/tilt_bins_<slug>.csv         — full per-bin per-class metrics
  <output>/tilt_bins_<slug>.png         — ΔF1 vs tilt bin plot

Usage
-----
    python3 evaluate-stn-tilt-bins.py
    python3 evaluate-stn-tilt-bins.py --precision int8
    python3 evaluate-stn-tilt-bins.py --images path/to/img --labels path/to/lbl
    python3 evaluate-stn-tilt-bins.py --bins 0 20 45 70 90
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
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
DEFAULT_OUTPUT  = str(SCRIPT_DIR / "benchmark_results/stn-tilt-bins")
CLASSES         = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# Default bin edges in degrees (from-horizontal tilt, inclusive lower / exclusive upper)
DEFAULT_BINS = [0, 10, 30, 60, 80, 90]


# ---------------------------------------------------------------------------
# Shared infrastructure (mirrored from evaluate-stn-bypass.py)
# ---------------------------------------------------------------------------
def _load_eval_module():
    path = SCRIPT_DIR / "evaluate-fomo.py"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find evaluate-fomo.py at {path}")
    spec = importlib.util.spec_from_file_location("evaluate_fomo", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        if not _patch_bypass_stn(m):
            raise RuntimeError(f"No GridSample node in {model_path}")
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
    try:
        input_size = int(sess.get_inputs()[0].shape[2])
    except (TypeError, ValueError):
        input_size = 480
    return sess, input_size


# ---------------------------------------------------------------------------
# Tilt angle from OBB label
# ---------------------------------------------------------------------------
def _gate_tilt(label_path: Path) -> float | None:
    """Return the tilt-from-horizontal of the railroad-crossing OBB, or None.

    Tilt is in [0°, 90°]: 0 = horizontal, 90 = vertical.
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
            tilt  = min(angle, 180 - angle)   # fold to [0°, 90°]
            tilts.append(tilt)
    except (FileNotFoundError, ValueError):
        pass
    return max(tilts) if tilts else None


def _bin_index(tilt: float, edges: list[float]) -> int:
    """Return 0-based bin index for `tilt` given bin edges."""
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if tilt < hi or i == len(edges) - 2:   # last bin is inclusive at top
            return i
    return len(edges) - 2   # should not reach


def _bin_label(edges: list[float], i: int) -> str:
    return f"{edges[i]:.0f}–{edges[i+1]:.0f}°"


# ---------------------------------------------------------------------------
# Per-bin evaluation loop
# ---------------------------------------------------------------------------
def run_binned_evaluation(
    sess: ort.InferenceSession,
    input_size: int,
    images_dir: str,
    labels_dir: str,
    classes: list[str],
    ev,
    bin_edges: list[float],
    confmat_conf: float,
    confmat_dist: float,
    label: str = "",
) -> list[dict]:
    """Run inference once, accumulate stats into per-tilt-bin buckets.

    Returns a list of result dicts (one per bin), in the same format as
    evaluate-stn-bypass.py::run_evaluation().
    Images with no class-0 annotation are placed in the nearest-horizontal bin.
    Images where the class-0 annotation falls in no bin (should not happen) are
    placed in the last bin.
    """
    n_bins = len(bin_edges) - 1
    in_name = sess.get_inputs()[0].name
    nc      = len(classes)
    ndist   = len(ev.DISTANCE_THRESHOLDS)
    primary_ti = int(np.argmin(np.abs(ev.DISTANCE_THRESHOLDS - ev.PRIMARY_DISTANCE)))

    # Per-bin accumulators
    stats_tp   = [[[] for _ in range(nc)] for _ in range(n_bins)]
    stats_conf = [[[] for _ in range(nc)] for _ in range(n_bins)]
    n_gt       = [np.zeros(nc, dtype=np.int64) for _ in range(n_bins)]
    n_images   = [0] * n_bins
    inf_times  = []

    img_paths = sorted(
        p for p in Path(images_dir).iterdir()
        if p.suffix.lower() in ev.IMG_EXTS
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    total = len(img_paths)
    for done, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        tilt = _gate_tilt(label_path)

        # Assign to bin — images with no gate go to bin 0 (horizontal)
        b = _bin_index(tilt if tilt is not None else 0.0, bin_edges)
        n_images[b] += 1

        gts = [g for g in ev.load_gt(str(label_path)) if 0 <= g[0] < nc]

        blob = ev.preprocess(img, input_size)
        t0   = time.perf_counter()
        out  = sess.run(None, {in_name: blob})[0]
        inf_times.append(time.perf_counter() - t0)

        dets = ev.postprocess(out, nc, ev.CONF_THRESHOLD_INFER)
        dets.sort(key=lambda d: -d["score"])

        for gc, _ in gts:
            n_gt[b][gc] += 1

        tp_arr = ev.match_for_map(dets, gts, ndist)
        for di, d in enumerate(dets):
            stats_tp[b][d["cls"]].append(tp_arr[di])
            stats_conf[b][d["cls"]].append(d["score"])

        if done % 50 == 0 or done == total:
            print(f"  [{done:>4}/{total}] {label}")

    # Compute per-bin metrics
    results = []
    for b in range(n_bins):
        ap     = np.zeros((nc, ndist), dtype=np.float64)
        p_best = np.zeros(nc)
        r_best = np.zeros(nc)
        f1_best = np.zeros(nc)

        for c in range(nc):
            if not stats_tp[b][c] or n_gt[b][c] == 0:
                continue
            tp   = np.array(stats_tp[b][c], dtype=bool)
            conf = np.array(stats_conf[b][c], dtype=np.float64)
            order = np.argsort(-conf)
            tp   = tp[order]
            conf = conf[order]
            fp   = ~tp
            tp_cum = tp.cumsum(0)
            fp_cum = fp.cumsum(0)
            recall    = tp_cum / max(n_gt[b][c], 1)
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

        results.append({
            "bin_idx":          b,
            "bin_label":        _bin_label(bin_edges, b),
            "n_images":         n_images[b],
            "classes":          classes,
            "n_gt":             n_gt[b],
            "ap":               ap,
            "ap_primary":       float(ap[:, primary_ti].mean()),
            "distance_thresholds": ev.DISTANCE_THRESHOLDS,
            "primary_ti":       primary_ti,
            "primary_distance": float(ev.DISTANCE_THRESHOLDS[primary_ti]),
            "precision":        p_best,
            "recall":           r_best,
            "f1":               f1_best,
        })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_bin_table(bins_active: list[dict], bins_bypass: list[dict], classes: list[str]):
    """Print a per-bin table focused on railroad-crossing (class 0)."""
    rc_idx = 0  # railroad-crossing

    header = (
        f"\n{'─'*100}\n"
        f"  {'Tilt bin':<22}  {'n':>5}  "
        f"{'[A] P':>8}  {'[A] R':>8}  {'[A] F1':>8}  {'[A] AP':>8}  "
        f"{'[B] P':>8}  {'[B] R':>8}  {'[B] F1':>8}  {'[B] AP':>8}  "
        f"{'ΔP':>8}  {'ΔR':>8}  {'ΔF1':>8}  {'ΔAP':>8}\n"
        f"  {'[A]=STN-active  [B]=STN-bypass  ·  railroad-crossing only':<95}\n"
        f"{'─'*100}"
    )
    print(header)

    for ra, rb in zip(bins_active, bins_bypass):
        pa,  ra_r,  f1a, apa = (
            ra["precision"][rc_idx], ra["recall"][rc_idx],
            ra["f1"][rc_idx],        ra["ap"][rc_idx, ra["primary_ti"]],
        )
        pb,  rb_r,  f1b, apb = (
            rb["precision"][rc_idx], rb["recall"][rc_idx],
            rb["f1"][rc_idx],        rb["ap"][rc_idx, rb["primary_ti"]],
        )
        marker = " ◀" if abs(f1a - f1b) == max(abs(r["f1"][rc_idx] - s["f1"][rc_idx])
                                                 for r, s in zip(bins_active, bins_bypass)) else ""
        print(
            f"  {ra['bin_label']:<22}  {ra['n_images']:>5}  "
            f"  {pa:8.4f}  {ra_r:8.4f}  {f1a:8.4f}  {apa:8.4f}  "
            f"  {pb:8.4f}  {rb_r:8.4f}  {f1b:8.4f}  {apb:8.4f}  "
            f"  {pa-pb:+8.4f}  {ra_r-rb_r:+8.4f}  {f1a-f1b:+8.4f}  {apa-apb:+8.4f}"
            f"{marker}"
        )

    print(f"{'─'*100}")

    # Summary for all classes
    print(f"\n  Mean F1 across all classes (active vs bypass per bin):")
    print(f"  {'Tilt bin':<22}  {'n':>5}  {'[A] mean F1':>12}  {'[B] mean F1':>12}  {'ΔF1':>10}")
    for ra, rb in zip(bins_active, bins_bypass):
        print(
            f"  {ra['bin_label']:<22}  {ra['n_images']:>5}  "
            f"  {ra['f1'].mean():12.4f}  {rb['f1'].mean():12.4f}  "
            f"  {ra['f1'].mean()-rb['f1'].mean():+10.4f}"
        )
    print()


def print_insights(bins_active: list[dict], bins_bypass: list[dict], classes: list[str]):
    rc_idx = 0
    deltas = [ra["f1"][rc_idx] - rb["f1"][rc_idx]
              for ra, rb in zip(bins_active, bins_bypass)]
    best_bin = int(np.argmax(deltas))
    worst_bin = int(np.argmin(deltas))

    print("─" * 72)
    print("  INSIGHTS")
    print("─" * 72)

    print(f"\n  railroad-crossing ΔF1 across tilt bins:")
    for i, (ra, rb, d) in enumerate(zip(bins_active, bins_bypass, deltas)):
        bar_len = int(abs(d) * 200)
        bar = ("▓" if d >= 0 else "░") * min(bar_len, 30)
        print(f"    {ra['bin_label']:<18}  Δ={d:+.4f}  {bar}")

    if deltas[best_bin] > 0.01:
        print(f"\n  Largest STN benefit: {bins_active[best_bin]['bin_label']} "
              f"(ΔF1 = {deltas[best_bin]:+.4f})")
        print(f"  → The STN correction matters most when the gate is in this angular range.")
    else:
        print(f"\n  No tilt bin shows a meaningful STN benefit (max ΔF1 = {deltas[best_bin]:+.4f}).")

    if deltas[worst_bin] < -0.01:
        print(f"\n  The STN slightly hurts in {bins_bypass[worst_bin]['bin_label']} "
              f"(ΔF1 = {deltas[worst_bin]:+.4f}).")
        print(f"  → Consider whether the rotation correction over-compensates at this angle.")

    # Count images per bin
    total = sum(ra["n_images"] for ra in bins_active)
    print(f"\n  Dataset composition ({total} images with class-0 labels):")
    for ra in bins_active:
        pct = 100.0 * ra["n_images"] / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"    {ra['bin_label']:<18}  {ra['n_images']:>4} images  ({pct:4.1f}%)  {bar}")

    print("─" * 72)


def save_bin_csv(path: str, bins_active: list[dict], bins_bypass: list[dict], classes: list[str]):
    rows = []
    for mode_label, bins in [("active", bins_active), ("bypass", bins_bypass)]:
        for b in bins:
            for ci, cls in enumerate(classes):
                rows.append({
                    "bin":       b["bin_label"],
                    "n_images":  b["n_images"],
                    "mode":      mode_label,
                    "class":     cls,
                    "precision": b["precision"][ci],
                    "recall":    b["recall"][ci],
                    "f1":        b["f1"][ci],
                    "ap":        b["ap"][ci, b["primary_ti"]],
                    "n_gt":      b["n_gt"][ci],
                })
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  [OK] CSV → {path}")


def save_bin_plot(path: str, bins_active: list[dict], bins_bypass: list[dict],
                  classes: list[str], precision: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    rc_idx = 0
    bin_labels = [ra["bin_label"] for ra in bins_active]
    n_bins = len(bin_labels)
    n_images = [ra["n_images"] for ra in bins_active]

    f1_active  = [ra["f1"][rc_idx] for ra in bins_active]
    f1_bypass  = [rb["f1"][rc_idx] for rb in bins_bypass]
    deltas_rc  = [a - b for a, b in zip(f1_active, f1_bypass)]

    # Also compute mean-all-class deltas
    deltas_mean = [ra["f1"].mean() - rb["f1"].mean()
                   for ra, rb in zip(bins_active, bins_bypass)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(n_bins)

    # ── Panel 1: F1 per bin (railroad-crossing) ─────────────────────────────
    ax = axes[0]
    w = 0.35
    bars_a = ax.bar(x - w/2, f1_active, w, label="STN-active",  color="#2563EB", alpha=0.85)
    bars_b = ax.bar(x + w/2, f1_bypass, w, label="STN-bypass",  color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("F1")
    ax.set_title("railroad-crossing F1 per tilt bin", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar in bars_a:
        v = bar.get_height()
        if v > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars_b:
        v = bar.get_height()
        if v > 0.02:
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    # secondary axis: image count
    ax2 = ax.twinx()
    ax2.plot(x, n_images, "k--o", markersize=5, linewidth=1.2, alpha=0.5, label="n images")
    ax2.set_ylabel("# images", color="gray", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="gray")

    # ── Panel 2: ΔF1 (railroad-crossing) per bin ────────────────────────────
    ax = axes[1]
    colours = ["#16A34A" if d >= 0 else "#DC2626" for d in deltas_rc]
    bars = ax.bar(x, deltas_rc, color=colours, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("ΔF1  (active − bypass)")
    ax.set_title("ΔF1 (railroad-crossing) vs gate tilt", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for xi, d in zip(x, deltas_rc):
        if abs(d) > 0.001:
            ax.text(xi, d + (0.003 if d >= 0 else -0.003),
                    f"{d:+.4f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=9, fontweight="bold")

    # ── Panel 3: ΔF1 mean-all-classes per bin ───────────────────────────────
    ax = axes[2]
    colours_m = ["#16A34A" if d >= 0 else "#DC2626" for d in deltas_mean]
    ax.bar(x, deltas_mean, color=colours_m, alpha=0.75, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("ΔF1  (active − bypass)")
    ax.set_title("ΔF1 (mean all classes) vs gate tilt", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for xi, d in zip(x, deltas_mean):
        if abs(d) > 0.001:
            ax.text(xi, d + (0.003 if d >= 0 else -0.003),
                    f"{d:+.4f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=9, fontweight="bold")

    fig.suptitle(
        f"STN-FOMO {precision.upper()} — active vs bypass stratified by gate tilt angle",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] plot → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--precision", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--images",  default=DEFAULT_IMAGES)
    ap.add_argument("--labels",  default=DEFAULT_LABELS)
    ap.add_argument("--output",  default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--bins", nargs="+", type=float, default=DEFAULT_BINS,
        metavar="DEG",
        help="Bin edges in degrees (e.g. --bins 0 10 30 60 80 90). "
             "Must be strictly increasing and start at 0, end at 90.",
    )
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    ap.add_argument("--threads",   type=int, default=4)
    ap.add_argument("--confmat-conf", type=float, default=0.25)
    ap.add_argument("--confmat-dist", type=float, default=0.10)
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
    print(f"\nSTN-FOMO tilt-bin ablation  ·  {args.precision.upper()}")
    print(f"Bins: {' | '.join(bin_names)}")
    print(f"Images: {args.images}")

    # Pre-scan: show bin distribution
    img_paths = sorted(
        p for p in Path(args.images).iterdir()
        if p.suffix.lower() in ev.IMG_EXTS
    )
    counts = [0] * n_bins
    for img_path in img_paths:
        lp = Path(args.labels) / (img_path.stem + ".txt")
        tilt = _gate_tilt(lp)
        b = _bin_index(tilt if tilt is not None else 0.0, bin_edges)
        counts[b] += 1
    print("Pre-scan bin distribution:")
    for i, (name, cnt) in enumerate(zip(bin_names, counts)):
        print(f"  {name:<18}  {cnt:>4} images")
    print()

    # Build sessions
    print("[1/4] Loading STN-active session …")
    sess_active, input_size = _make_session(
        model_path, bypass=False,
        providers=args.providers, num_threads=args.threads,
    )
    print("[2/4] Loading STN-bypass session …")
    sess_bypass, _ = _make_session(
        model_path, bypass=True,
        providers=args.providers, num_threads=args.threads,
    )

    # Run binned evaluation
    print(f"\n[3/4] Evaluating STN-active …")
    bins_active = run_binned_evaluation(
        sess_active, input_size, args.images, args.labels, CLASSES, ev,
        bin_edges, args.confmat_conf, args.confmat_dist, label="STN-active",
    )

    print(f"\n[4/4] Evaluating STN-bypass …")
    bins_bypass = run_binned_evaluation(
        sess_bypass, input_size, args.images, args.labels, CLASSES, ev,
        bin_edges, args.confmat_conf, args.confmat_dist, label="STN-bypass",
    )

    # Report
    print_bin_table(bins_active, bins_bypass, CLASSES)
    print_insights(bins_active, bins_bypass, CLASSES)

    # Save
    slug = f"stn-fomo-{args.precision}"
    save_bin_csv(
        os.path.join(args.output, f"tilt_bins_{slug}.csv"),
        bins_active, bins_bypass, CLASSES,
    )
    save_bin_plot(
        os.path.join(args.output, f"tilt_bins_{slug}.png"),
        bins_active, bins_bypass, CLASSES, args.precision,
    )
    print(f"\nArtifacts written to: {args.output}")


if __name__ == "__main__":
    main()
