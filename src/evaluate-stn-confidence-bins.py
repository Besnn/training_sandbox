#!/usr/bin/env python3
"""STN-FOMO: per-sample railroad-crossing confidence, STN-active vs STN-bypass,
stratified by gate tilt bin.

This is the per-sample, confidence-based counterpart to evaluate-stn-tilt-bins.py
(which compares full-image P/R/F1/AP via detection matching). Here, instead of
running the full detection-matching pipeline, we read the railroad-crossing
confidence directly at the ground-truth gate location (3x3-cell max around the
GT centroid in the output heatmap) for two model configurations:

  [A] STN-active   - the unmodified model; the STN applies whatever rotation
                       it learned for this image.
  [B] STN-bypass   - the GridSample's grid is replaced with the identity grid,
                       i.e. no geometric normalization at all.

No synthetic/injected rotation angles are involved (cf.
evaluate-stn-orientation-sweep.py, which sweeps a constant injected angle).

For fp32, the localization network's actual learned (t1, t2) ~= (cos a, sin a)
is also extracted via the `linear_1` tensor, giving the rotation angle a the
STN applied for that image - reported per bin as a reference only (does not
affect [A]/[B]).

Outputs
-------
  stdout                                      - per-bin summary table + insights
  <output>/confidence_bins_<slug>.csv         - per-sample confidence (active, bypass, gain)
  <output>/confidence_bins_<slug>.png         - 2-panel figure:
                                                  (1) mean confidence per bin, active vs bypass
                                                  (2) histogram of per-sample gain (active-bypass)

Usage
-----
    python3 evaluate-stn-confidence-bins.py
    python3 evaluate-stn-confidence-bins.py --precision int8 --no-loc
    python3 evaluate-stn-confidence-bins.py --bins 0 20 45 70 90
    python3 evaluate-stn-confidence-bins.py --max-per-bin 30
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
DEFAULT_OUTPUT  = str(SCRIPT_DIR / "benchmark_results/stn-confidence-bins")
CLASSES         = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# Same tilt bins as evaluate-stn-tilt-bins.py
DEFAULT_BINS = [-5, 10, 30, 60, 80, 89, 92]
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


def _grid_hw_from_gridsample(model, gs_node):
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
    return H, W


def _patch_bypass_stn(model: onnx.ModelProto) -> bool:
    """Replace the GridSample's grid with the identity grid (= no STN warp)."""
    gs_node = next((n for n in model.graph.node if n.op_type == "GridSample"), None)
    if gs_node is None:
        return False
    H, W = _grid_hw_from_gridsample(model, gs_node)
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
                   bypass: bool = False,
                   loc_tensor: str | None = None) -> tuple[ort.InferenceSession, int, bool]:
    """Build a session.

    bypass=False -> unmodified model (learned STN active).
    bypass=True  -> STN replaced with the identity grid (no geometric normalization).
    loc_tensor   -> if given (and not bypass), try to also expose this tensor
                     as an output.
    Returns (session, input_size, has_loc_output).
    """
    m = onnx.load(model_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass

    if bypass:
        if not _patch_bypass_stn(m):
            raise RuntimeError(f"No GridSample node in {model_path}")

    has_loc = False
    if loc_tensor and not bypass:
        has_loc = _add_localization_output(m, loc_tensor)

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
    return sess, input_size, has_loc


# ---------------------------------------------------------------------------
# Gate tilt / binning  (mirrors evaluate-stn-tilt-bins.py)
# ---------------------------------------------------------------------------
def _gate_samples(label_path: Path) -> list[tuple[float, float, float]]:
    """Return [(unsigned_tilt_deg, cx, cy), ...] for every class-0 OBB.

    unsigned_tilt in [0, 90]: 0 = horizontal, 90 = vertical.
    (cx, cy) is the OBB centroid, normalised [0,1].
    """
    out = []
    try:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9 or parts[0] != "0":
                continue
            coords = list(map(float, parts[1:9]))
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            tilt  = min(angle, 180 - angle)
            cx = sum(coords[0::2]) / 4.0
            cy = sum(coords[1::2]) / 4.0
            out.append((tilt, cx, cy))
    except (FileNotFoundError, ValueError):
        pass
    return out


def _bin_index(tilt: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if tilt < hi or i == len(edges) - 2:
            return i
    return len(edges) - 2


def _bin_label(edges: list[float], i: int) -> str:
    return f"{edges[i]:.0f}–{edges[i+1]:.0f}°"


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# ---------------------------------------------------------------------------
# Confidence read-out
# ---------------------------------------------------------------------------
def _prob_map_class0(raw, nc: int, ev) -> np.ndarray:
    """Return the (H, W) railroad-crossing probability map from raw model output."""
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected output shape {arr.shape}")
    if arr.shape[0] not in (nc, nc + 1):
        if arr.shape[-1] in (nc, nc + 1):
            arr = arr.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unexpected channel count in shape {arr.shape}")
    if arr.shape[0] == nc + 1:
        probs = ev.softmax(arr, axis=0)
        return probs[1]   # channel 0 = background, channel 1 = class 0
    probs = ev.sigmoid(arr)
    return probs[0]


def _confidence_at(prob_map: np.ndarray, cx: float, cy: float) -> float:
    h, w = prob_map.shape
    col = min(max(int(cx * w), 0), w - 1)
    row = min(max(int(cy * h), 0), h - 1)
    y0, y1 = max(0, row - 1), min(h, row + 2)
    x0, x1 = max(0, col - 1), min(w, col + 2)
    return float(prob_map[y0:y1, x0:x1].max())


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_bin_table(bin_names: list[str], bin_stats: list[dict | None]):
    print("\n" + "─" * 96)
    print(f"  {'Tilt bin':<10}  {'n':>5}  {'conf[A]':>9}  {'conf[B]':>9}  {'gain':>9}  "
          f"{'%impr':>7}  {'%wors':>7}  {'mean learned α':>15}")
    print(f"  [A]=STN-active  [B]=STN-bypass  ·  confidence at GT (railroad-crossing, 3x3 max)")
    print("─" * 96)
    for name, bs in zip(bin_names, bin_stats):
        if bs is None:
            print(f"  {name:<10}  {0:>5}   (no samples)")
            continue
        learned_str = f"{bs['alpha_mean']:+.2f}°" if bs["alpha_mean"] is not None else "n/a"
        print(
            f"  {name:<10}  {bs['n']:>5}  "
            f"{bs['conf_active_mean']:>9.4f}  {bs['conf_bypass_mean']:>9.4f}  "
            f"{bs['gain_mean']:>+9.4f}  "
            f"{100*bs['frac_improved']:>6.1f}%  {100*bs['frac_worsened']:>6.1f}%  "
            f"{learned_str:>15}"
        )
    print("─" * 96)


def print_insights(bin_names: list[str], bin_stats: list[dict | None]):
    print("\n" + "─" * 72)
    print("  INSIGHTS")
    print("─" * 72)

    valid = [(n, bs) for n, bs in zip(bin_names, bin_stats) if bs is not None]
    if not valid:
        print("  (no samples)")
        print("─" * 72)
        return

    print("\n  Mean STN gain (active − bypass) across tilt bins:")
    for name, bs in valid:
        d = bs["gain_mean"]
        bar_len = int(abs(d) * 400)
        bar = ("▓" if d >= 0 else "░") * min(bar_len, 30)
        print(f"    {name:<10}  Δconf={d:+.4f}  {bar}")

    gains = {n: bs["gain_mean"] for n, bs in valid}
    best_name  = max(gains, key=gains.get)
    worst_name = min(gains, key=gains.get)

    if gains[best_name] > 0.005:
        print(f"\n  Largest STN benefit: {best_name} (Δconf = {gains[best_name]:+.4f})")
        print(f"  → The STN's learned warp most increases head confidence at this tilt.")
    else:
        print(f"\n  No tilt bin shows a meaningful STN benefit "
              f"(max Δconf = {gains[best_name]:+.4f}).")

    if gains[worst_name] < -0.005:
        print(f"\n  The STN slightly hurts in {worst_name} (Δconf = {gains[worst_name]:+.4f}).")

    total = sum(bs["n"] for _, bs in valid)
    print(f"\n  Dataset composition ({total} railroad-crossing annotations):")
    for name, bs in valid:
        pct = 100.0 * bs["n"] / max(total, 1)
        bar = "█" * int(pct / 2)
        print(f"    {name:<10}  {bs['n']:>4} samples  ({pct:4.1f}%)  {bar}")

    print("─" * 72)


def print_correlations(r_tilt_gain: float, r_dist_gain: float, n: int):
    print("\n" + "─" * 72)
    print("  CORRELATION: STN gain (active − bypass) vs gate orientation")
    print(f"  (real test-set images only, n={n} -- no synthetic rotation)")
    print("─" * 72)
    print(f"  corr(tilt, gain)                = {r_tilt_gain:+.3f}")
    print(f"  corr(dist-from-canonical, gain) = {r_dist_gain:+.3f}")
    print(f"\n  dist-from-canonical = min(tilt, 90-tilt)  "
          f"(0° = horizontal/vertical, 45° = diagonal)")
    if r_dist_gain > 0.1:
        print(f"\n  -> Positive correlation: the STN's active-vs-bypass benefit tends to grow")
        print(f"     as the gate's natural orientation moves away from the dataset's two")
        print(f"     dominant (canonical) poses -- consistent with the STN performing more")
        print(f"     useful geometric normalization for off-canonical gates, using only")
        print(f"     REAL test-set images (no synthetic rotation needed).")
    elif r_dist_gain < -0.1:
        print(f"\n  -> Negative correlation: the STN's benefit is largest for gates ALREADY")
        print(f"     near a canonical pose, and shrinks (or reverses) for off-canonical gates.")
    else:
        print(f"\n  -> No meaningful linear correlation between gate orientation and STN gain")
        print(f"     in the real test-set distribution (|r| <= 0.1).")
    print("─" * 72)


def save_csv(path: str, samples: list[dict], bin_names: list[str],
              conf_active: np.ndarray, conf_bypass: np.ndarray,
              gain: np.ndarray, alpha_learned: list[float | None],
              dist_canon: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "image", "bin", "tilt_deg", "dist_from_canonical_deg",
                    "alpha_learned_deg", "conf_active", "conf_bypass", "gain"])
        for i, s in enumerate(samples):
            w.writerow([
                i, s["img"].name, bin_names[s["bin"]], f"{s['tilt']:.2f}",
                f"{dist_canon[i]:.2f}",
                "" if alpha_learned[i] is None else f"{alpha_learned[i]:.2f}",
                f"{conf_active[i]:.5f}", f"{conf_bypass[i]:.5f}", f"{gain[i]:.5f}",
            ])
    print(f"  [OK] CSV -> {path}")


def save_plot(path: str, bin_names: list[str], bin_stats: list[dict | None],
               samples: list[dict], gain: np.ndarray, precision: str,
               dist_canon: np.ndarray, r_dist_gain: float):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plot)")
        return

    n_bins = len(bin_names)
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, n_bins))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # ── Panel 1: mean confidence per bin, active vs bypass ──────────────────
    ax = axes[0]
    x = np.arange(n_bins)
    w = 0.35
    means_a = np.array([bs["conf_active_mean"] if bs else 0.0 for bs in bin_stats])
    stds_a  = np.array([bs["conf_active_std"]  if bs else 0.0 for bs in bin_stats])
    means_b = np.array([bs["conf_bypass_mean"] if bs else 0.0 for bs in bin_stats])
    stds_b  = np.array([bs["conf_bypass_std"]  if bs else 0.0 for bs in bin_stats])
    n_samp  = [bs["n"] if bs else 0 for bs in bin_stats]

    ax.bar(x - w/2, means_a, w, yerr=stds_a, capsize=3,
           label="STN-active", color="#2563EB", alpha=0.85)
    ax.bar(x + w/2, means_b, w, yerr=stds_b, capsize=3,
           label="STN-bypass", color="#DC2626", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("railroad-crossing confidence at GT (3x3 max)")
    ax.set_title("Confidence at GT: STN-active vs STN-bypass", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(x, n_samp, "k--o", markersize=5, linewidth=1.2, alpha=0.5)
    ax2.set_ylabel("# samples", color="gray", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="gray")

    # ── Panel 2: histogram of per-sample gain, by bin ───────────────────────
    ax = axes[1]
    bin_idx_arr = np.array([s["bin"] for s in samples])
    if len(gain):
        pad = max(float(np.max(np.abs(gain))), 0.01) * 1.1
    else:
        pad = 0.01
    bin_edges_hist = np.linspace(-pad, pad, 31)
    for b in range(n_bins):
        g = gain[bin_idx_arr == b]
        if len(g) == 0:
            continue
        ax.hist(g, bins=bin_edges_hist, alpha=0.45, color=colours[b],
                label=f"{bin_names[b]} (n={len(g)})", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("gain  (conf[active] − conf[bypass])")
    ax.set_ylabel("# samples")
    ax.set_title("Per-sample STN effect on confidence", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 3: gain vs dist-from-canonical, with regression line ──────────
    ax = axes[2]
    ax.scatter(dist_canon, gain, s=14, alpha=0.35, color="#2563EB",
               edgecolor="none", label="sample")
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    if len(dist_canon) >= 2 and np.std(dist_canon) > 0:
        m, b = np.polyfit(dist_canon, gain, 1)
        xs = np.linspace(float(dist_canon.min()), float(dist_canon.max()), 50)
        ax.plot(xs, m * xs + b, color="#DC2626", linewidth=2.0,
                label=f"fit: r={r_dist_gain:+.3f}")
    ax.set_xlabel("dist-from-canonical = min(tilt, 90−tilt)  (deg)")
    ax.set_ylabel("gain  (conf[active] − conf[bypass])")
    ax.set_title("STN gain vs gate orientation (real images)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"STN-FOMO {precision.upper()} — confidence at GT, active vs bypass, by gate tilt",
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
    ap.add_argument("--max-per-bin", type=int, default=0,
                    help="Cap samples per tilt bin (0 = no cap).")
    ap.add_argument("--no-loc", action="store_true",
                    help="Skip the learned-angle reference readout.")
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
    nc = len(CLASSES)
    n_bins = len(bin_edges) - 1
    bin_names = [_bin_label(bin_edges, i) for i in range(n_bins)]

    print(f"\nSTN-FOMO confidence bins  ·  {args.precision.upper()}")
    print(f"Tilt bins: {' | '.join(bin_names)}")

    # -------------------------------------------------------------------
    # Gather samples
    # -------------------------------------------------------------------
    img_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in ev.IMG_EXTS
    )
    samples = []   # dicts: img_path, bin_idx, cx, cy, tilt
    bin_counts_all = [0] * n_bins
    for img_path in img_paths:
        label_path = Path(args.labels) / (img_path.stem + ".txt")
        for tilt, cx, cy in _gate_samples(label_path):
            b = _bin_index(tilt, bin_edges)
            bin_counts_all[b] += 1
            samples.append({"img": img_path, "bin": b, "cx": cx, "cy": cy, "tilt": tilt})

    if args.max_per_bin > 0:
        per_bin: dict[int, list] = {}
        for s in samples:
            per_bin.setdefault(s["bin"], []).append(s)
        capped = []
        for b, lst in per_bin.items():
            step = max(1, len(lst) // args.max_per_bin)
            capped.extend(lst[::step][: args.max_per_bin])
        samples = capped

    bin_counts = [0] * n_bins
    for s in samples:
        bin_counts[s["bin"]] += 1

    print(f"\nSamples per bin (railroad-crossing annotations):")
    for name, all_n, used_n in zip(bin_names, bin_counts_all, bin_counts):
        suffix = "" if all_n == used_n else f"  (capped from {all_n})"
        print(f"  {name:<10}  {used_n:>4}{suffix}")
    print(f"  TOTAL: {len(samples)} samples")

    # -------------------------------------------------------------------
    # Cache images
    # -------------------------------------------------------------------
    print("\nCaching images …")
    img_cache: dict[Path, np.ndarray] = {}
    for s in samples:
        if s["img"] not in img_cache:
            img = cv2.imread(str(s["img"]))
            if img is not None:
                img_cache[s["img"]] = img
    samples = [s for s in samples if s["img"] in img_cache]
    n = len(samples)

    # -------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------
    loc_tensor = None if args.no_loc else args.loc_tensor
    print("\n[1/2] Loading STN-active session …")
    sess_active, in_sz_active, has_loc = _make_session(
        model_path, args.providers, args.threads, bypass=False, loc_tensor=loc_tensor,
    )
    if loc_tensor and not has_loc:
        print(f"  (tensor '{loc_tensor}' not found in graph — learned-angle readout disabled)")

    print("[2/2] Loading STN-bypass session …")
    sess_bypass, in_sz_bypass, _ = _make_session(
        model_path, args.providers, args.threads, bypass=True,
    )

    # -------------------------------------------------------------------
    # Pass 1: STN-active
    # -------------------------------------------------------------------
    conf_active  = np.zeros(n, dtype=np.float64)
    conf_bypass  = np.zeros(n, dtype=np.float64)
    alpha_learned: list[float | None] = [None] * n

    print(f"\nRunning STN-active on {n} samples …")
    in_name = sess_active.get_inputs()[0].name
    loc_idx = None
    if has_loc:
        out_names = [o.name for o in sess_active.get_outputs()]
        loc_idx = out_names.index(loc_tensor)
    for i, s in enumerate(samples):
        blob = ev.preprocess(img_cache[s["img"]], in_sz_active)
        outs = sess_active.run(None, {in_name: blob})
        pmap = _prob_map_class0(outs[0], nc, ev)
        conf_active[i] = _confidence_at(pmap, s["cx"], s["cy"])
        if loc_idx is not None:
            t1, t2 = outs[loc_idx][0]
            alpha_learned[i] = math.degrees(math.atan2(float(t2), float(t1)))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}]")

    # -------------------------------------------------------------------
    # Pass 2: STN-bypass
    # -------------------------------------------------------------------
    print(f"\nRunning STN-bypass on {n} samples …")
    in_name_b = sess_bypass.get_inputs()[0].name
    for i, s in enumerate(samples):
        blob = ev.preprocess(img_cache[s["img"]], in_sz_bypass)
        out = sess_bypass.run(None, {in_name_b: blob})[0]
        pmap = _prob_map_class0(out, nc, ev)
        conf_bypass[i] = _confidence_at(pmap, s["cx"], s["cy"])
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}]")

    gain = conf_active - conf_bypass

    # -------------------------------------------------------------------
    # Correlation: STN gain vs gate orientation (real images, no synthetic
    # rotation)
    # -------------------------------------------------------------------
    tilts = np.array([s["tilt"] for s in samples], dtype=np.float64)
    dist_canon = np.minimum(tilts, 90.0 - tilts)
    r_tilt_gain = _pearson(tilts, gain)
    r_dist_gain = _pearson(dist_canon, gain)

    # -------------------------------------------------------------------
    # Aggregate per bin
    # -------------------------------------------------------------------
    bin_stats: list[dict | None] = []
    for b in range(n_bins):
        idx = [i for i, s in enumerate(samples) if s["bin"] == b]
        if not idx:
            bin_stats.append(None)
            continue
        ca = conf_active[idx]
        cb = conf_bypass[idx]
        g  = gain[idx]
        learned = [alpha_learned[i] for i in idx if alpha_learned[i] is not None]
        bin_stats.append({
            "n":                len(idx),
            "conf_active_mean": float(ca.mean()),
            "conf_active_std":  float(ca.std()),
            "conf_bypass_mean": float(cb.mean()),
            "conf_bypass_std":  float(cb.std()),
            "gain_mean":        float(g.mean()),
            "gain_std":         float(g.std()),
            "frac_improved":    float(np.mean(g > 1e-9)),
            "frac_worsened":    float(np.mean(g < -1e-9)),
            "alpha_mean":       float(np.mean(learned)) if learned else None,
        })

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    print_bin_table(bin_names, bin_stats)
    print_insights(bin_names, bin_stats)
    print_correlations(r_tilt_gain, r_dist_gain, n)

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
    slug = f"stn-fomo-{args.precision}"
    save_csv(
        os.path.join(args.output, f"confidence_bins_{slug}.csv"),
        samples, bin_names, conf_active, conf_bypass, gain, alpha_learned, dist_canon,
    )
    save_plot(
        os.path.join(args.output, f"confidence_bins_{slug}.png"),
        bin_names, bin_stats, samples, gain, args.precision, dist_canon, r_dist_gain,
    )
    print(f"\nArtifacts written to: {args.output}")


if __name__ == "__main__":
    main()
