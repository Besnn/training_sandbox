#!/usr/bin/env python3
"""Plot per-class precision-recall curves for FOMO / STN-FOMO ONNX models.

For every model it sweeps the per-detection confidence, matches detections to
ground-truth centroids at a single normalized distance, and draws one PR curve
per class with matplotlib. Each class panel is annotated with its BEST confidence
(the threshold that maximizes F1) plus that operating point's P / R / F1 and the
class AP. One PNG is written per model.

Decoding mirrors evaluate-fomo.py and auto-detects the head from the channel
count:
  - num_classes+1 channels -> legacy softmax head (background = channel 0)
  - num_classes   channels -> sigmoid heatmap head (CenterNet / STN-FOMO)

Usage:
    python3 plot_pr_curves_fomo.py
    python3 plot_pr_curves_fomo.py \
        --model "FOMO-STN=models/stn-fomo-480-onnx/stn-fomo-480.onnx" \
        --images <dir> --labels <dir> --dist 0.05
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import label as nd_label, maximum_filter

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
# Classes trained with Gaussian target smearing get sub-cell peak refinement;
# rigid hard-centroid classes (and STN-FOMO, which is fully rigid) do not.
SMEARED_CLASSES = (0,)
DEFAULT_MODELS = {
    "FOMO-STN-FP32": "models/stn-fomo-480-onnx/stn-fomo-480.onnx",
    "FOMO-STN-INT8": "models/stn-fomo-480-onnx/stn-fomo-480-int8.onnx",
    "FOMO-FP32": "models/fomo-480-onnx/fomo-480.onnx",
    "FOMO-INT8": "models/fomo-480-onnx/fomo-480-int8.onnx",
}
DEFAULT_VAL_IMAGES = str(SCRIPT_DIR / "datasets/yolo_pl_test/images")
DEFAULT_VAL_LABELS = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "benchmark_results/pr-curves/")

# Matching distance (normalized image units) and the loose confidence floor used
# to collect detections for the sweep.
PRIMARY_DISTANCE = 0.047
CONF_FLOOR = 0.01


# --------------------------------------------------------------------------- #
# Inference + decode (mirrors evaluate-fomo.py)
# --------------------------------------------------------------------------- #
def preprocess(image, input_size):
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return np.expand_dims(blob, 0)


def softmax(logits, axis):
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _decode_softmax_percell(arr, conf_thres):
    probs = softmax(arr, axis=0)
    pred_map = probs.argmax(axis=0)
    score_map = probs.max(axis=0)
    h, w = score_map.shape
    dets = []
    active = (pred_map != 0) & (score_map >= conf_thres)
    for y, x in zip(*np.where(active)):
        dets.append({
            "cell": ((int(x) + 0.5) / w, (int(y) + 0.5) / h),
            "cls": int(pred_map[y, x] - 1),
            "score": float(score_map[y, x]),
        })
    return dets


def _decode_heatmap_peaks(arr, conf_thres, smeared_classes=SMEARED_CLASSES):
    probs = sigmoid(arr)
    pred_map = probs.argmax(axis=0)
    score_map = probs.max(axis=0)
    h, w = score_map.shape

    mx = maximum_filter(score_map, size=3, mode="constant", cval=0.0)
    peak_mask = (score_map == mx) & (score_map >= conf_thres)
    if not peak_mask.any():
        return []
    labeled, n_comp = nd_label(peak_mask, structure=np.ones((3, 3), dtype=np.int32))

    dets = []
    for comp_id in range(1, n_comp + 1):
        ys, xs = np.where(labeled == comp_id)
        pi = int(np.argmax(score_map[ys, xs]))
        py, px = int(ys[pi]), int(xs[pi])
        cls = int(pred_map[py, px])
        if cls in smeared_classes:
            y0, y1 = max(0, py - 1), min(h, py + 2)
            x0, x1 = max(0, px - 1), min(w, px + 2)
            win = score_map[y0:y1, x0:x1]
            wsum = win.sum()
            gy, gx = np.mgrid[y0:y1, x0:x1]
            cy = float((win * gy).sum() / wsum) if wsum > 0 else float(py)
            cx = float((win * gx).sum() / wsum) if wsum > 0 else float(px)
        else:
            cy, cx = float(py), float(px)
        dets.append({
            "cell": ((cx + 0.5) / w, (cy + 0.5) / h),
            "cls": cls,
            "score": float(score_map[py, px]),
        })
    return dets


def postprocess(raw, num_classes, conf_thres):
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected FOMO output shape: {arr.shape}")
    if arr.shape[0] not in (num_classes, num_classes + 1):
        if arr.shape[-1] in (num_classes, num_classes + 1):
            arr = arr.transpose(2, 0, 1)
        else:
            raise ValueError(
                f"FOMO output has {arr.shape[0]} channels, expected "
                f"{num_classes} (heatmap) or {num_classes + 1} (softmax)"
            )
    if arr.shape[0] == num_classes + 1:
        return _decode_softmax_percell(arr, conf_thres)
    return _decode_heatmap_peaks(arr, conf_thres)


# --------------------------------------------------------------------------- #
# Ground truth + matching
# --------------------------------------------------------------------------- #
def load_gt(label_path):
    """Return [(class_id, (x, y))] normalized; centroid or 4-corner polygon."""
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            cls = int(p[0])
            if len(p) == 9:
                x = sum(float(p[i]) for i in (1, 3, 5, 7)) / 4.0
                y = sum(float(p[i]) for i in (2, 4, 6, 8)) / 4.0
            else:
                x, y = float(p[1]), float(p[2])
            out.append((cls, (x, y)))
    return out


def match_image(dets, gts, dist_thres, num_classes):
    """Greedy-by-score, same-class match within dist_thres.

    Returns (records, n_gt) where records is a list of (cls, score, is_tp) for
    every detection and n_gt counts ground-truth centroids per class.
    """
    n_gt = np.zeros(num_classes, dtype=np.int64)
    for gc, _ in gts:
        if 0 <= gc < num_classes:
            n_gt[gc] += 1

    used = [False] * len(gts)
    records = []
    for d in sorted(dets, key=lambda d: -d["score"]):
        c = d["cls"]
        if not (0 <= c < num_classes):
            continue
        best_i, best_dist = -1, dist_thres
        for i, (gc, (gx, gy)) in enumerate(gts):
            if used[i] or gc != c:
                continue
            dist = ((d["cell"][0] - gx) ** 2 + (d["cell"][1] - gy) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist, best_i = dist, i
        if best_i >= 0:
            used[best_i] = True
            records.append((c, d["score"], True))
        else:
            records.append((c, d["score"], False))
    return records, n_gt


# --------------------------------------------------------------------------- #
# PR curve math
# --------------------------------------------------------------------------- #
def compute_ap(recall, precision):
    """101-point interpolated AP (COCO-style)."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def pr_curve(scores, tps, n_gt):
    """Cumulative PR curve plus the best-F1 operating point.

    Returns a dict with recall, precision, ap, and best_{conf,p,r,f1}.
    """
    empty = {"recall": np.array([]), "precision": np.array([]), "ap": 0.0,
             "best_conf": float("nan"), "best_p": 0.0, "best_r": 0.0,
             "best_f1": 0.0, "n_gt": int(n_gt), "n_det": len(scores)}
    if len(scores) == 0 or n_gt == 0:
        return empty

    scores = np.asarray(scores, dtype=np.float64)
    tps = np.asarray(tps, dtype=bool)
    order = np.argsort(-scores)
    scores, tps = scores[order], tps[order]

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(~tps)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-16)
    bi = int(np.argmax(f1))
    return {
        "recall": recall, "precision": precision,
        "ap": compute_ap(recall, precision),
        "best_conf": float(scores[bi]), "best_p": float(precision[bi]),
        "best_r": float(recall[bi]), "best_f1": float(f1[bi]),
        "n_gt": int(n_gt), "n_det": len(scores),
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model_path, images_dir, labels_dir, providers, classes,
             dist_thres, conf_floor):
    sess = ort.InferenceSession(model_path, providers=providers)
    inp = sess.get_inputs()[0]
    in_name = inp.name
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 480

    nc = len(classes)
    cls_scores = [[] for _ in range(nc)]
    cls_tps = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, dtype=np.int64)

    img_paths = sorted(p for p in Path(images_dir).iterdir()
                       if p.suffix.lower() in IMG_EXTS)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gts = [g for g in load_gt(str(Path(labels_dir) / (img_path.stem + ".txt")))
               if 0 <= g[0] < nc]
        out = sess.run(None, {in_name: preprocess(img, input_size)})[0]
        dets = postprocess(out, nc, conf_floor)
        records, img_n_gt = match_image(dets, gts, dist_thres, nc)
        n_gt += img_n_gt
        for c, score, is_tp in records:
            cls_scores[c].append(score)
            cls_tps[c].append(is_tp)

    return {c: pr_curve(cls_scores[c], cls_tps[c], n_gt[c]) for c in range(nc)}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_pr(name, results, classes, dist_thres, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(classes)
    cols = 2
    rows = (nc + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.0 * rows),
                             squeeze=False)
    fig.suptitle(f"{name} — precision-recall per class (dist ≤ {dist_thres} norm)",
                 fontsize=15, fontweight="bold")

    for c, cname in enumerate(classes):
        ax = axes[c // cols][c % cols]
        r = results[c]
        ax.set_title(cname, fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        if r["recall"].size == 0:
            ax.text(0.5, 0.5, "no data\n(no GT or no detections)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        ax.plot(r["recall"], r["precision"], color="#1F4E79", lw=2,
                label=f"AP={r['ap']:.3f}")
        # Best-F1 operating point.
        ax.plot(r["best_r"], r["best_p"], "o", color="#D7263D", ms=9, zorder=5)

        txt = (f"best conf = {r['best_conf']:.3f}\n"
               f"P={r['best_p']:.3f}  R={r['best_r']:.3f}\n"
               f"F1={r['best_f1']:.3f}   AP={r['ap']:.3f}\n"
               f"GT={r['n_gt']}  det={r['n_det']}")
        ax.text(0.03, 0.06, txt, transform=ax.transAxes, fontsize=10,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="#FFF4E6", ec="#D7263D", alpha=0.95))
        ax.legend(loc="upper right", fontsize=9)

    # Hide any unused panels.
    for k in range(nc, rows * cols):
        axes[k // cols][k % cols].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=DEFAULT_VAL_IMAGES)
    ap.add_argument("--labels", default=DEFAULT_VAL_LABELS)
    ap.add_argument("--model", action="append", default=None, metavar="NAME=PATH",
                    help="Model entry (repeatable). Defaults to the bundled models.")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--dist", type=float, default=PRIMARY_DISTANCE,
                    help="Match distance in normalized image units.")
    ap.add_argument("--conf-floor", type=float, default=CONF_FLOOR,
                    help="Loose confidence floor for collecting detections.")
    return ap.parse_args()


def parse_models(items):
    out = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--model expects NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def resolve_model_path(path):
    p = Path(path)
    if p.is_absolute() or p.is_file():
        return str(p)
    rel = SCRIPT_DIR / p
    return str(rel if rel.is_file() else p)


def main():
    args = parse_args()
    models = parse_models(args.model) if args.model else dict(DEFAULT_MODELS)

    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")
    os.makedirs(args.output_dir, exist_ok=True)

    ran = False
    for name, mp in models.items():
        model_path = resolve_model_path(mp)
        if not os.path.isfile(model_path):
            print(f"[WARN] {name}: model not found at {mp} — skipping")
            continue
        print(f"\n=== {name} ({model_path}) ===")
        results = evaluate(model_path, args.images, args.labels, args.providers,
                           args.classes, args.dist, args.conf_floor)
        for c, cname in enumerate(args.classes):
            r = results[c]
            print(f"  {cname:<22} best_conf={r['best_conf']:.3f}  "
                  f"P={r['best_p']:.3f} R={r['best_r']:.3f} F1={r['best_f1']:.3f} "
                  f"AP={r['ap']:.3f}  (GT={r['n_gt']}, det={r['n_det']})")

        slug = name.lower().replace("/", "_").replace(" ", "_")
        out_path = os.path.join(args.output_dir, f"{slug}_pr_curves.png")
        plot_pr(name, results, args.classes, args.dist, out_path)
        print(f"[OK] {name}: PR curves written to {out_path}")
        ran = True

    if not ran:
        raise SystemExit("No models evaluated.")


if __name__ == "__main__":
    main()
