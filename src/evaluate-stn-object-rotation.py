#!/usr/bin/env python3
"""STN-FOMO: object-centric rotation sweep.

Instead of rotating the whole 480x480 frame (evaluate-stn-rotation.py, where
objects rotate out of frame and BOTH STN-active and STN-bypass collapse for
the same trivial reason), this script rotates ONLY the railroad-crossing gate
itself, in place, and leaves the rest of the scene untouched.

Method
------
For every railroad-crossing (class 0) annotation in the test set:
  1. Compute the gate's OBB centroid (cx, cy) in pixel space and a square
     patch size = ceil(diagonal of the OBB's axis-aligned bbox * --margin).
     The diagonal guarantees the gate stays fully inside the patch under ANY
     rotation angle.
  2. For each angle theta in the sweep, build a synthetic image: copy the
     original image and replace the size x size square centered at (cx, cy)
     with that SAME region rotated in place by theta. The patch is sampled
     from a slightly larger surrounding crop so the rotated content fully
     fills the patch -- no black borders, even at +/-90 degrees. Everything
     outside the patch (other objects, background, the rest of the scene) is
     pixel-for-pixel identical to the original image.
  3. The GT centroid does NOT move (rotation is about its own center), so no
     label remapping is needed -- theta=0 reproduces the original image
     exactly.
  4. Run STN-active and STN-bypass on the synthetic image and read the
     railroad-crossing confidence at the GT centroid (3x3 max).

This isolates exactly what the STN was designed for: does locally
re-orienting the gate change the head's confidence, and does the STN's
learned warp recover more of that confidence than a no-op (bypass) does --
for gates that start in EVERY tilt bin, not just the mid-arc ones?

Outputs
-------
  stdout                                    - overall + per-bin summary tables,
                                              insights
  <output>/object_rotation_<slug>.csv       - per-sample-per-angle data
  <output>/object_rotation_<slug>.png       - 2-panel figure:
                                              (1) overall confidence vs theta,
                                                  active vs bypass (+ std band)
                                              (2) per-original-tilt-bin gain
                                                  (active - bypass) vs theta
  <output>/examples/*.png                   - (optional, --save-examples N)
                                              before/after patch grids for the
                                              first N samples

Usage
-----
    python3 evaluate-stn-object-rotation.py
    python3 evaluate-stn-object-rotation.py --angles -60 -30 0 30 60
    python3 evaluate-stn-object-rotation.py --max-images 60
    python3 evaluate-stn-object-rotation.py --save-examples 5

Runtime
-------
With defaults (13 angles x ~370 gate annotations x 2 sessions) this is
roughly ~9600 inferences on CPU -- a few minutes. Use --angles or
--max-images to speed this up.
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
DEFAULT_OUTPUT  = str(SCRIPT_DIR / "benchmark_results/stn-object-rotation")
CLASSES         = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# Original-tilt bins (gate angle from horizontal, folded to [0,90])
DEFAULT_BINS   = [0, 10, 30, 60, 80, 90]
# Injected per-object rotation sweep (degrees). theta=0 == original image.
DEFAULT_ANGLES = list(range(-90, 91, 15))
# Patch size = diagonal of the OBB's axis-aligned bbox * margin.
DEFAULT_MARGIN = 1.15


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


def _make_session(model_path: str, providers: list[str], num_threads: int,
                   bypass: bool = False) -> tuple[ort.InferenceSession, int]:
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

    sess = ort.InferenceSession(m.SerializeToString(), sess_options=opts, providers=configured)
    try:
        input_size = int(sess.get_inputs()[0].shape[2])
    except (TypeError, ValueError):
        input_size = 480
    return sess, input_size


# ---------------------------------------------------------------------------
# Gate geometry
# ---------------------------------------------------------------------------
def _gate_samples_px(label_path: Path, img_w: int, img_h: int,
                       margin: float) -> list[dict]:
    """Return per-class-0-OBB dicts with pixel-space crop params.

    Each dict: cx_px, cy_px (pixel centroid), size (square patch side, px),
    cx_n, cy_n (normalised centroid, for confidence read-out / GT), tilt
    (deg, [0,90], folded -- same definition as evaluate-stn-tilt-bins.py).
    """
    out = []
    try:
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 9 or parts[0] != "0":
                continue
            coords = list(map(float, parts[1:9]))
            xs_n = coords[0::2]
            ys_n = coords[1::2]
            cx_n = sum(xs_n) / 4.0
            cy_n = sum(ys_n) / 4.0

            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
            tilt  = min(angle, 180 - angle)

            xs_px = [x * img_w for x in xs_n]
            ys_px = [y * img_h for y in ys_n]
            cx_px = sum(xs_px) / 4.0
            cy_px = sum(ys_px) / 4.0
            bw = max(xs_px) - min(xs_px)
            bh = max(ys_px) - min(ys_px)
            diag = math.hypot(bw, bh)
            size = max(8, int(math.ceil(diag * margin)))

            out.append({
                "cx_px": cx_px, "cy_px": cy_px, "size": size,
                "cx_n": cx_n, "cy_n": cy_n, "tilt": tilt,
            })
    except (FileNotFoundError, ValueError):
        pass
    return out


def _rotate_object_inplace(img: np.ndarray, cx: float, cy: float, size: int,
                             angle_deg: float) -> np.ndarray:
    """Return a copy of `img` with the size x size square centered at (cx,cy)
    replaced by that same region rotated in place by `angle_deg` about its
    own center. Sampled from a larger surrounding crop (edge-replicated near
    image borders) so the result has no black borders. theta=0 is a pixel-
    exact no-op.
    """
    h, w = img.shape[:2]
    outer = int(math.ceil(size * math.sqrt(2))) + 4

    cxi, cyi = int(round(cx)), int(round(cy))

    pad = outer
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    ox0 = (cxi + pad) - outer // 2
    oy0 = (cyi + pad) - outer // 2
    outer_crop = padded[oy0:oy0 + outer, ox0:ox0 + outer]

    if angle_deg % 360 == 0:
        rotated = outer_crop
    else:
        center = (outer / 2.0 - 0.5, outer / 2.0 - 0.5)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        rotated = cv2.warpAffine(outer_crop, M, (outer, outer),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    off = (outer - size) // 2
    inner = rotated[off:off + size, off:off + size]

    out = img.copy()
    ix0, iy0 = cxi - size // 2, cyi - size // 2
    ix1, iy1 = ix0 + size, iy0 + size

    sx0, sy0 = max(0, -ix0), max(0, -iy0)
    sx1 = size - max(0, ix1 - w)
    sy1 = size - max(0, iy1 - h)
    dx0, dy0 = max(0, ix0), max(0, iy0)
    dx1, dy1 = min(w, ix1), min(h, iy1)

    if dx1 > dx0 and dy1 > dy0 and sx1 > sx0 and sy1 > sy0:
        out[dy0:dy1, dx0:dx1] = inner[sy0:sy1, sx0:sx1]
    return out


def _bin_index(tilt: float, edges: list[float]) -> int:
    for i in range(len(edges) - 1):
        hi = edges[i + 1]
        if tilt < hi or i == len(edges) - 2:
            return i
    return len(edges) - 2


def _bin_label(edges: list[float], i: int) -> str:
    return f"{edges[i]:.0f}–{edges[i+1]:.0f}°"


# ---------------------------------------------------------------------------
# Confidence read-out
# ---------------------------------------------------------------------------
def _prob_map_class0(raw, nc: int, ev) -> np.ndarray:
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
        return probs[1]
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
def print_overall_table(angles, conf_active_mean, conf_bypass_mean, gain_mean, n):
    print("\n" + "─" * 72)
    print(f"  OVERALL (n={n} gate annotations, all tilt bins pooled)")
    print("─" * 72)
    print(f"  {'theta':>7}  {'conf[A]':>9}  {'conf[B]':>9}  {'gain':>9}")
    for ai, theta in enumerate(angles):
        marker = "  <- theta=0 (original image)" if theta == 0 else ""
        print(f"  {theta:>+7.0f}  {conf_active_mean[ai]:>9.4f}  "
              f"{conf_bypass_mean[ai]:>9.4f}  {gain_mean[ai]:>+9.4f}{marker}")
    print("─" * 72)


def print_bin_table(angles, bin_names, bin_n, gain_per_bin):
    print("\n" + "─" * 110)
    print("  Per-original-tilt-bin STN gain (active - bypass) vs injected rotation theta")
    header = f"  {'Tilt bin':<10}  {'n':>4}  " + "  ".join(f"{t:>+7.0f}" for t in angles)
    print(header)
    print("─" * 110)
    zero_idx = int(np.argmin(np.abs(np.array(angles))))
    for name, n, row in zip(bin_names, bin_n, gain_per_bin):
        if n == 0:
            print(f"  {name:<10}  {0:>4}   (no samples)")
            continue
        print(f"  {name:<10}  {n:>4}  " + "  ".join(f"{g:>+7.4f}" for g in row))
    print("─" * 110)


def print_insights(angles, bin_names, bin_n, gain_per_bin):
    print("\n" + "─" * 72)
    print("  INSIGHTS")
    print("─" * 72)
    angles_arr = np.array(angles, dtype=np.float64)
    zero_idx = int(np.argmin(np.abs(angles_arr)))

    for name, n, row in zip(bin_names, bin_n, gain_per_bin):
        if n == 0:
            continue
        row = np.asarray(row)
        gain0 = row[zero_idx]
        best_idx = int(np.argmax(np.abs(row)))
        best_theta = angles_arr[best_idx]
        best_gain = row[best_idx]
        if best_idx != zero_idx and abs(best_gain) > abs(gain0) + 0.01:
            print(f"  {name:<10} (n={n:>4}): gain at theta=0 is {gain0:+.4f}; "
                  f"grows to {best_gain:+.4f} at theta={best_theta:+.0f}°")
            print(f"             -> the STN's benefit over bypass INCREASES when this "
                  f"bin's gates are artificially rotated by {best_theta:+.0f}°.")
        else:
            print(f"  {name:<10} (n={n:>4}): gain stays ~flat across theta "
                  f"(theta=0: {gain0:+.4f}, max|gain|: {best_gain:+.4f} @ {best_theta:+.0f}°)")
    print("─" * 72)


def save_csv(path: str, samples: list[dict], bin_names: list[str], angles: list[float],
              conf_active: np.ndarray, conf_bypass: np.ndarray):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "image", "bin", "tilt_deg", "theta_deg",
                    "conf_active", "conf_bypass", "gain"])
        for i, s in enumerate(samples):
            for ai, theta in enumerate(angles):
                ca = conf_active[i, ai]
                cb = conf_bypass[i, ai]
                w.writerow([i, s["img_path"].name, bin_names[s["bin"]],
                            f"{s['tilt']:.2f}", f"{theta:.1f}",
                            f"{ca:.5f}", f"{cb:.5f}", f"{ca-cb:.5f}"])
    print(f"  [OK] CSV -> {path}")


def save_plot(path: str, angles, bin_names, bin_n, gain_per_bin,
               conf_active_mean, conf_active_std, conf_bypass_mean, conf_bypass_std,
               precision: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available - skipping plot)")
        return

    angles_arr = np.array(angles, dtype=np.float64)
    n_bins = len(bin_names)
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, n_bins))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # -- Panel 1: overall confidence vs theta -------------------------------
    ax = axes[0]
    ax.plot(angles_arr, conf_active_mean, "-o", color="#2563EB", markersize=4,
            linewidth=1.8, label="STN-active")
    ax.fill_between(angles_arr, conf_active_mean - conf_active_std,
                     conf_active_mean + conf_active_std, color="#2563EB", alpha=0.12)
    ax.plot(angles_arr, conf_bypass_mean, "-o", color="#DC2626", markersize=4,
            linewidth=1.8, label="STN-bypass")
    ax.fill_between(angles_arr, conf_bypass_mean - conf_bypass_std,
                     conf_bypass_mean + conf_bypass_std, color="#DC2626", alpha=0.12)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6,
               label="theta=0 (original image)")
    ax.set_xlabel("injected per-object rotation theta (deg)")
    ax.set_ylabel("railroad-crossing confidence at GT (3x3 max)")
    ax.set_title("Object-centric rotation: overall confidence", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # -- Panel 2: per-bin gain vs theta --------------------------------------
    ax = axes[1]
    for b in range(n_bins):
        if bin_n[b] == 0:
            continue
        ax.plot(angles_arr, gain_per_bin[b], "-o", color=colours[b], markersize=4,
                linewidth=1.8, label=f"{bin_names[b]} (n={bin_n[b]})")
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("injected per-object rotation theta (deg)")
    ax.set_ylabel("gain = conf[active] - conf[bypass]")
    ax.set_title("STN benefit vs theta, by ORIGINAL gate-tilt bin", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"STN-FOMO {precision.upper()} - object-centric rotation sweep",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] plot -> {path}")


def save_examples(out_dir: str, samples: list[dict], img_cache: dict,
                    angles: list[float], n_examples: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available - skipping examples)")
        return

    os.makedirs(out_dir, exist_ok=True)
    n_examples = min(n_examples, len(samples))
    n_cols = len(angles)

    for i in range(n_examples):
        s = samples[i]
        img = img_cache[s["img_path"]]
        size = s["size"]
        view = int(size * 1.6)

        fig, axes = plt.subplots(1, n_cols, figsize=(2.0 * n_cols, 2.4))
        if n_cols == 1:
            axes = [axes]
        for ai, theta in enumerate(angles):
            synth = _rotate_object_inplace(img, s["cx_px"], s["cy_px"], size, theta)
            cxi, cyi = int(round(s["cx_px"])), int(round(s["cy_px"]))
            x0, y0 = max(0, cxi - view // 2), max(0, cyi - view // 2)
            x1, y1 = min(synth.shape[1], x0 + view), min(synth.shape[0], y0 + view)
            crop = synth[y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ax = axes[ai]
            ax.imshow(crop_rgb)
            # patch boundary, in crop-local coords
            rx0, ry0 = (cxi - size // 2) - x0, (cyi - size // 2) - y0
            ax.add_patch(plt.Rectangle((rx0, ry0), size, size, fill=False,
                                         edgecolor="#16A34A", linewidth=1.5))
            ax.set_title(f"theta={theta:+.0f}°" + ("\n(original)" if theta == 0 else ""),
                         fontsize=9)
            ax.axis("off")

        fig.suptitle(f"{s['img_path'].name}  (tilt={s['tilt']:.1f}°)", fontsize=10)
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"sample_{i:03d}_{s['img_path'].stem}.png")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    print(f"  [OK] {n_examples} example grid(s) -> {out_dir}")


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
                    help="Original gate-tilt bin edges in degrees (default: 0 10 30 60 80 90).")
    ap.add_argument("--angles", nargs="+", type=float, default=DEFAULT_ANGLES,
                    help="Injected per-object rotation angles to sweep, in degrees "
                         "(default: -90..90 step 15). 0 == original image.")
    ap.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                    help="Patch size = ceil(OBB bbox diagonal * margin) (default: 1.15).")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Cap the number of gate annotations processed (0 = no cap).")
    ap.add_argument("--save-examples", type=int, default=0,
                    help="Save before/after patch grids for the first N samples.")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    bin_edges = sorted(args.bins)
    angles = sorted(args.angles)
    if len(bin_edges) < 2:
        raise SystemExit("--bins needs at least 2 edge values")
    if not angles:
        raise SystemExit("--angles needs at least 1 value")

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

    print(f"\nSTN-FOMO object-centric rotation sweep  ·  {args.precision.upper()}")
    print(f"Original-tilt bins: {' | '.join(bin_names)}")
    print(f"Angle sweep: {angles}")

    # -------------------------------------------------------------------
    # Gather samples (one per class-0 OBB)
    # -------------------------------------------------------------------
    img_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in ev.IMG_EXTS
    )
    samples = []
    img_cache: dict[Path, np.ndarray] = {}
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = Path(args.labels) / (img_path.stem + ".txt")
        gates = _gate_samples_px(label_path, w, h, args.margin)
        if not gates:
            continue
        for g in gates:
            b = _bin_index(g["tilt"], bin_edges)
            samples.append({"img_path": img_path, "bin": b, **g})
        img_cache[img_path] = img

    if args.max_images > 0:
        samples = samples[: args.max_images]
        img_cache = {p: img for p, img in img_cache.items()
                      if any(s["img_path"] == p for s in samples)}

    bin_counts = [0] * n_bins
    for s in samples:
        bin_counts[s["bin"]] += 1

    print(f"\nGate annotations per original-tilt bin:")
    for name, cnt in zip(bin_names, bin_counts):
        print(f"  {name:<10}  {cnt:>4}")
    n = len(samples)
    print(f"  TOTAL: {n} samples, {len(angles)} angles, 2 sessions "
          f"=> {n * len(angles) * 2} inferences")

    # -------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------
    print("\n[1/2] Loading STN-active session …")
    sess_active, in_sz = _make_session(model_path, args.providers, args.threads, bypass=False)
    print("[2/2] Loading STN-bypass session …")
    sess_bypass, _ = _make_session(model_path, args.providers, args.threads, bypass=True)

    in_name_a = sess_active.get_inputs()[0].name
    in_name_b = sess_bypass.get_inputs()[0].name

    # -------------------------------------------------------------------
    # Sweep
    # -------------------------------------------------------------------
    conf_active = np.zeros((n, len(angles)), dtype=np.float64)
    conf_bypass = np.zeros((n, len(angles)), dtype=np.float64)

    print(f"\nRunning sweep on {n} samples …")
    for i, s in enumerate(samples):
        img = img_cache[s["img_path"]]
        for ai, theta in enumerate(angles):
            synth = _rotate_object_inplace(img, s["cx_px"], s["cy_px"], s["size"], theta)

            blob_a = ev.preprocess(synth, in_sz)
            out_a = sess_active.run(None, {in_name_a: blob_a})[0]
            pmap_a = _prob_map_class0(out_a, nc, ev)
            conf_active[i, ai] = _confidence_at(pmap_a, s["cx_n"], s["cy_n"])

            blob_b = ev.preprocess(synth, in_sz)
            out_b = sess_bypass.run(None, {in_name_b: blob_b})[0]
            pmap_b = _prob_map_class0(out_b, nc, ev)
            conf_bypass[i, ai] = _confidence_at(pmap_b, s["cx_n"], s["cy_n"])

        if (i + 1) % 25 == 0 or (i + 1) == n:
            print(f"  [{i+1:>4}/{n}]")

    gain = conf_active - conf_bypass

    # -------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------
    conf_active_mean = conf_active.mean(axis=0)
    conf_active_std  = conf_active.std(axis=0)
    conf_bypass_mean = conf_bypass.mean(axis=0)
    conf_bypass_std  = conf_bypass.std(axis=0)
    gain_mean        = gain.mean(axis=0)

    bin_idx_arr = np.array([s["bin"] for s in samples]) if n else np.array([], dtype=int)
    gain_per_bin = []
    for b in range(n_bins):
        mask = bin_idx_arr == b
        if not np.any(mask):
            gain_per_bin.append(np.zeros(len(angles)))
            continue
        gain_per_bin.append(gain[mask].mean(axis=0))

    # -------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------
    print_overall_table(angles, conf_active_mean, conf_bypass_mean, gain_mean, n)
    print_bin_table(angles, bin_names, bin_counts, gain_per_bin)
    print_insights(angles, bin_names, bin_counts, gain_per_bin)

    # -------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------
    slug = f"stn-fomo-{args.precision}"
    save_csv(
        os.path.join(args.output, f"object_rotation_{slug}.csv"),
        samples, bin_names, angles, conf_active, conf_bypass,
    )
    save_plot(
        os.path.join(args.output, f"object_rotation_{slug}.png"),
        angles, bin_names, bin_counts, gain_per_bin,
        conf_active_mean, conf_active_std, conf_bypass_mean, conf_bypass_std,
        args.precision,
    )
    if args.save_examples > 0:
        save_examples(
            os.path.join(args.output, "examples"),
            samples, img_cache, angles, args.save_examples,
        )
    print(f"\nArtifacts written to: {args.output}")


if __name__ == "__main__":
    main()
