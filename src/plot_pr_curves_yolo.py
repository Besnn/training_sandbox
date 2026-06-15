#!/usr/bin/env python3
"""Plot per-class precision-recall curves for YOLOv8-OBB / YOLO26-OBB ONNX models.

For every model it sweeps the per-detection confidence, matches detections to
ground-truth oriented boxes by polygon IoU at a single IoU threshold (default
0.5, the mAP50 operating point), and draws one PR curve per class with
matplotlib. Each class panel is annotated with its BEST confidence (the
threshold that maximizes F1) plus that operating point's P / R / F1 and the
class AP. One PNG is written per model.

Decoding mirrors evaluate-yolo-obb.py and supports both ONNX export layouts:
  - Raw grid (YOLOv8 default):  [cx, cy, w, h, *class_scores, angle]
  - End-to-end / NMS-free:      [cx, cy, w, h, conf, cls_id, angle]

YOLO-OBB label format (one box per line, normalized):
    class_id x1 y1 x2 y2 x3 y3 x4 y4

Usage:
    python3 plot_pr_curves_yolo.py
    python3 plot_pr_curves_yolo.py \
        --model "YOLOv8m-OBB=models/yolov8m-obb-onnx/yolov8m-obb-fp32.onnx" \
        --images <dir> --labels <dir> --iou 0.5
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
DEFAULT_MODELS = {
    # "YOLOv8m-OBB-fp32": "models/yolov8m-obb-onnx/yolov8m-obb-fp32.onnx",
    # "YOLOv8n-OBB-fp32": "models/yolov8n-obb-onnx/yolov8n-obb-fp32.onnx",
    # "YOLOv8n-OBB-int8": "models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx",
    # "YOLO26n-OBB-fp32": "models/yolo26n-obb-onnx/yolo26n-obb-fp32.onnx",
    "YOLO26n-OBB-int8": "models/yolo26n-obb-onnx/yolo26n-obb-int8.onnx",
}
DEFAULT_VAL_IMAGES = str(SCRIPT_DIR / "datasets/yolo_pl_test/images")
DEFAULT_VAL_LABELS = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "benchmark_results/pr-curves/")

# Match IoU for P/R/F1/AP (mAP50 operating point) and the loose confidence
# floor used to collect detections for the sweep (mirrors Ultralytics val).
PRIMARY_IOU = 0.5
CONF_FLOOR = 0.001
NMS_IOU = 0.1

# Minimum confidence for the deployable operating point. Each panel highlights
# the part of the curve at conf >= MIN_CONF, and a single global confidence
# (>= MIN_CONF, shared by all classes) that maximizes mean F1 is marked too.
MIN_CONF = 0.5


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def poly_iou(poly_a, poly_b):
    """IoU between two 4-vertex convex polygons (pixel coords)."""
    a = cv2.convexHull(poly_a.astype(np.float32))
    b = cv2.convexHull(poly_b.astype(np.float32))
    inter, _ = cv2.intersectConvexConvex(a, b)
    if inter <= 0:
        return 0.0
    area_a = cv2.contourArea(a)
    area_b = cv2.contourArea(b)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# Inference + decode (mirrors evaluate-yolo-obb.py)
# --------------------------------------------------------------------------- #
def preprocess(image, input_size):
    """Letterbox image like Ultralytics, preserving aspect ratio."""
    img_h, img_w = image.shape[:2]
    gain = min(input_size / img_w, input_size / img_h)
    new_w, new_h = int(round(img_w * gain)), int(round(img_h * gain))
    pad_w = (input_size - new_w) / 2
    pad_h = (input_size - new_h) / 2

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    meta = {"gain": gain, "pad_w": left, "pad_h": top}
    return np.expand_dims(blob, 0).astype(np.float32) / 255.0, meta


def xywhr_to_poly(cx, cy, w, h, angle):
    """Convert YOLO OBB xywhr arrays to 4-point polygons."""
    cos, sin = np.cos(angle), np.sin(angle)
    wx, wy = (w / 2) * cos, (w / 2) * sin
    hx, hy = -(h / 2) * sin, (h / 2) * cos
    return np.stack(
        [
            np.stack([cx + wx + hx, cy + wy + hy], axis=1),
            np.stack([cx + wx - hx, cy + wy - hy], axis=1),
            np.stack([cx - wx - hx, cy - wy - hy], axis=1),
            np.stack([cx - wx + hx, cy - wy + hy], axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def rotated_nms(dets, iou_thres):
    keep = []
    for c in np.unique([d["cls"] for d in dets]):
        idxs = [i for i, d in enumerate(dets) if d["cls"] == c]
        idxs.sort(key=lambda i: -dets[i]["score"])
        while idxs:
            best = idxs.pop(0)
            keep.append(best)
            idxs = [
                i for i in idxs
                if poly_iou(dets[best]["poly"], dets[i]["poly"]) <= iou_thres
            ]
    return keep


def postprocess(raw, input_size, img_w, img_h, num_classes, conf_thres, meta):
    """Decode YOLO-OBB head (raw grid or end-to-end NMS-free)."""
    preds = np.squeeze(raw)
    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T
    if preds.ndim != 2 or preds.shape[1] < 7:
        return []

    raw_grid_cols = 4 + num_classes + 1
    if preds.shape[1] >= raw_grid_cols:
        cxcywh = preds[:, :4]
        class_scores = preds[:, 4 : 4 + num_classes]
        angles = preds[:, 4 + num_classes]
        scores = class_scores.max(axis=1)
        cls_ids = class_scores.argmax(axis=1)
        end2end = False
    elif preds.shape[1] == 7:
        cxcywh = preds[:, :4]
        scores = preds[:, 4]
        cls_ids = preds[:, 5].astype(np.int32)
        angles = preds[:, 6]
        end2end = True
    else:
        return []

    coords_norm = np.nanmax(np.abs(cxcywh)) <= 2.0

    mask = scores > conf_thres
    if end2end:
        mask &= (cls_ids >= 0) & (cls_ids < num_classes)
    if not mask.any():
        return []
    cxcywh = cxcywh[mask]
    angles = angles[mask]
    scores = scores[mask]
    cls_ids = cls_ids[mask]

    if coords_norm:
        cxcywh[:, [0, 2]] *= input_size
        cxcywh[:, [1, 3]] *= input_size

    gain = meta["gain"]
    pad_w = meta["pad_w"]
    pad_h = meta["pad_h"]

    cx = (cxcywh[:, 0] - pad_w) / gain
    cy = (cxcywh[:, 1] - pad_h) / gain
    w = cxcywh[:, 2] / gain
    h = cxcywh[:, 3] / gain

    polys = xywhr_to_poly(cx, cy, w, h, angles)
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, img_w)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, img_h)

    dets = [
        {"poly": polys[i], "cls": int(cls_ids[i]), "score": float(scores[i])}
        for i in range(len(polys))
    ]
    if end2end:
        return dets
    keep = rotated_nms(dets, NMS_IOU)
    return [dets[i] for i in keep]


# --------------------------------------------------------------------------- #
# Ground truth + matching
# --------------------------------------------------------------------------- #
def load_gt(label_path, img_w, img_h):
    """Return [(class_id, polygon[4,2] in clipped pixel coords)]."""
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 9:
                continue
            try:
                cls = int(float(parts[0]))
                xy = np.array(parts[1:9], dtype=np.float64).reshape(4, 2)
            except ValueError:
                continue
            if not np.isfinite(xy).all():
                continue
            xy[:, 0] = np.clip(xy[:, 0], 0.0, 1.0) * img_w
            xy[:, 1] = np.clip(xy[:, 1], 0.0, 1.0) * img_h
            out.append((cls, xy.astype(np.float32)))
    return out


def match_image(dets, gts, iou_thres, num_classes):
    """Greedy-by-score, same-class IoU match.

    Returns (records, n_gt) where records is a list of (cls, score, is_tp) for
    every detection and n_gt counts ground-truth boxes per class.
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
        best_i, best_iou = -1, iou_thres
        for i, (gc, gp) in enumerate(gts):
            if used[i] or gc != c:
                continue
            iou = poly_iou(d["poly"], gp)
            if iou >= best_iou:
                best_iou, best_i = iou, i
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
    """Cumulative PR curve plus the best-F1 operating point."""
    empty = {"recall": np.array([]), "precision": np.array([]),
             "scores": np.array([]), "f1": np.array([]), "ap": 0.0,
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
        "recall": recall, "precision": precision, "scores": scores, "f1": f1,
        "ap": compute_ap(recall, precision),
        "best_conf": float(scores[bi]), "best_p": float(precision[bi]),
        "best_r": float(recall[bi]), "best_f1": float(f1[bi]),
        "n_gt": int(n_gt), "n_det": len(scores),
    }


def op_at_conf(r, conf):
    """P / R / F1 at the operating point conf >= `conf` (scores sorted desc).

    Returns None when no detection reaches `conf` or the class has no curve.
    """
    scores = r["scores"]
    if scores.size == 0:
        return None
    mask = scores >= conf
    if not mask.any():
        return None
    k = int(mask.sum()) - 1  # last (lowest-score) detection still >= conf
    return {"conf": float(conf), "p": float(r["precision"][k]),
            "r": float(r["recall"][k]), "f1": float(r["f1"][k])}


def global_best_conf(results, min_conf):
    """Single confidence >= min_conf maximizing mean F1 across scored classes.

    Returns (conf, mean_f1) or (None, 0.0) if nothing clears min_conf.
    """
    valid = [c for c, r in results.items()
             if r["n_gt"] > 0 and r["scores"].size > 0]
    cands = sorted({float(s) for c in valid for s in results[c]["scores"]
                    if s >= min_conf})
    if not valid or not cands:
        return None, 0.0
    best_t, best_mean = None, -1.0
    for t in cands:
        f1s = [(op_at_conf(results[c], t) or {"f1": 0.0})["f1"] for c in valid]
        m = float(np.mean(f1s))
        if m > best_mean:
            best_mean, best_t = m, t
    return best_t, best_mean


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(model_path, images_dir, labels_dir, providers, classes,
             iou_thres, conf_floor):
    sess = ort.InferenceSession(model_path, providers=providers)
    inp = sess.get_inputs()[0]
    in_name = inp.name
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 640

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
        h, w = img.shape[:2]
        gts = [g for g in load_gt(str(Path(labels_dir) / (img_path.stem + ".txt")), w, h)
               if 0 <= g[0] < nc]
        blob, meta = preprocess(img, input_size)
        out = sess.run(None, {in_name: blob})[0]
        dets = postprocess(out, input_size, w, h, nc, conf_floor, meta)
        records, img_n_gt = match_image(dets, gts, iou_thres, nc)
        n_gt += img_n_gt
        for c, score, is_tp in records:
            cls_scores[c].append(score)
            cls_tps[c].append(is_tp)

    return {c: pr_curve(cls_scores[c], cls_tps[c], n_gt[c]) for c in range(nc)}


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_pr(name, results, classes, iou_thres, out_path, min_conf, glob_conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nc = len(classes)
    cols = 2
    rows = (nc + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5.0 * rows),
                             squeeze=False)
    gtxt = f"{glob_conf:.3f}" if glob_conf is not None else "n/a"
    fig.suptitle(
        f"{name} — precision-recall per class (IoU ≥ {iou_thres})\n"
        f"global conf ≥ {min_conf:g} maximizing mean F1 = {gtxt}",
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
        # Highlight the conf >= min_conf segment (a leading prefix once the
        # curve is sorted by descending score).
        k = int((r["scores"] >= min_conf).sum())
        if k > 0:
            ax.plot(r["recall"][:k], r["precision"][:k], color="#2A9D8F",
                    lw=3.2, alpha=0.9, label=f"conf ≥ {min_conf:g}")
        # Best-F1 operating point.
        ax.plot(r["best_r"], r["best_p"], "o", color="#D7263D", ms=9, zorder=5)

        # Shared global-conf operating point for this class.
        gop = op_at_conf(r, glob_conf) if glob_conf is not None else None
        gline = "global conf: none ≥ floor"
        if gop is not None:
            ax.plot(gop["r"], gop["p"], "s", color="#2A9D8F", ms=9, zorder=6)
            gline = (f"@conf {gop['conf']:.3f}: P={gop['p']:.3f} "
                     f"R={gop['r']:.3f} F1={gop['f1']:.3f}")

        txt = (f"best conf = {r['best_conf']:.3f}\n"
               f"P={r['best_p']:.3f}  R={r['best_r']:.3f}\n"
               f"F1={r['best_f1']:.3f}   AP={r['ap']:.3f}\n"
               f"GT={r['n_gt']}  det={r['n_det']}\n"
               f"{gline}")
        ax.text(0.03, 0.06, txt, transform=ax.transAxes, fontsize=9,
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
    ap.add_argument("--iou", type=float, default=PRIMARY_IOU,
                    help="Match IoU threshold for TP/FP (mAP50 uses 0.5).")
    ap.add_argument("--conf-floor", type=float, default=CONF_FLOOR,
                    help="Loose confidence floor for collecting detections.")
    ap.add_argument("--min-conf", type=float, default=MIN_CONF,
                    help="Lower bound for the highlighted/global operating point.")
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
                           args.classes, args.iou, args.conf_floor)
        glob_conf, glob_mean_f1 = global_best_conf(results, args.min_conf)
        for c, cname in enumerate(args.classes):
            r = results[c]
            print(f"  {cname:<22} best_conf={r['best_conf']:.3f}  "
                  f"P={r['best_p']:.3f} R={r['best_r']:.3f} F1={r['best_f1']:.3f} "
                  f"AP={r['ap']:.3f}  (GT={r['n_gt']}, det={r['n_det']})")
        if glob_conf is not None:
            print(f"  -> global conf ≥ {args.min_conf:g} maximizing mean F1 = "
                  f"{glob_conf:.3f}  (mean F1={glob_mean_f1:.3f})")
        else:
            print(f"  -> no detection clears conf ≥ {args.min_conf:g}")

        slug = name.lower().replace("/", "_").replace(" ", "_")
        out_path = os.path.join(args.output_dir, f"{slug}_pr_curves.png")
        plot_pr(name, results, args.classes, args.iou, out_path,
                args.min_conf, glob_conf)
        print(f"[OK] {name}: PR curves written to {out_path}")
        ran = True

    if not ran:
        raise SystemExit("No models evaluated.")


if __name__ == "__main__":
    main()
