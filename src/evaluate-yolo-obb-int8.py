#!/usr/bin/env python3
"""Evaluate INT8 YOLO-OBB ONNX models.

Computes per-class precision, recall, F1, mAP@0.5, mAP@[0.5:0.95] and a
confusion matrix. Designed to run locally on macOS / aarch64 (Arduino Uno Q)
with just numpy, opencv and onnxruntime — no Ultralytics or torch needed.

YOLO-OBB label format (one box per line, normalized to image size):
    class_id x1 y1 x2 y2 x3 y3 x4 y4

Usage:
    python3 evaluate-yolo-obb-int8.py
    python3 evaluate-yolo-obb-int8.py \
        --images <dir> --labels <dir> \
        --model "YOLOv8-OBB-INT8=models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx" \
        --model "YOLO26-OBB-INT8=models/yolo26n-obb-onnx/yolo26n-obb-int8.onnx"
"""

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# COCO-style IoU sweep used for mAP@[.5:.95]
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
PERMISSIVE_IOU = 0.25

# Match Ultralytics val defaults: keep loose conf during AP collection
CONF_THRESHOLD_INFER = 0.001
NMS_IOU = 0.25

# Confusion matrix thresholds (mirror Ultralytics defaults)
CONFMAT_IOU = 0.25
CONFMAT_CONF = 0.6

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_MODELS = {
    "YOLOv8-OBB-INT8": "models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx",
}
DEFAULT_VAL_IMAGES = (
    "/Users/besnn/PycharmProjects/YOLOv8 Traffic Light Model/"
    "traffic_signal_detection/src/datasets/split_obb_dataset/test/images"
)
DEFAULT_VAL_LABELS = (
    "/Users/besnn/PycharmProjects/YOLOv8 Traffic Light Model/"
    "traffic_signal_detection/src/datasets/split_obb_dataset/test/labels"
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def poly_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
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
# I/O
# --------------------------------------------------------------------------- #
def load_gt(label_path: str, img_w: int, img_h: int):
    """Return list of (class_id, polygon[4,2] in pixel coords)."""
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            xy = np.array(parts[1:9], dtype=np.float64).reshape(4, 2)
            xy[:, 0] *= img_w
            xy[:, 1] *= img_h
            out.append((cls, xy.astype(np.float32)))
    return out


# --------------------------------------------------------------------------- #
# YOLO-OBB inference
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
    """Decode YOLO-OBB head.

    Supports two ONNX export layouts:
      - Raw grid (YOLOv8 default):  [cx, cy, w, h, *class_scores, angle]
      - End-to-end / NMS-free:      [cx, cy, w, h, conf, cls_id, angle]
        (Ultralytics end2end=True, e.g. YOLO26-OBB default export)
    """
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
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO-style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def match_for_thresholds(dets, gts, iou_thresholds):
    """Return (N_det, niou) bool TP array. Detections are already score-sorted."""
    tp = np.zeros((len(dets), len(iou_thresholds)), dtype=bool)
    if not dets or not gts:
        return tp
    iou_mat = np.zeros((len(dets), len(gts)), dtype=np.float32)
    for di, d in enumerate(dets):
        for gi, (gc, gp) in enumerate(gts):
            if d["cls"] == gc:
                iou_mat[di, gi] = poly_iou(d["poly"], gp)
    for ti, thr in enumerate(iou_thresholds):
        used = np.zeros(len(gts), dtype=bool)
        for di in range(len(dets)):
            best_iou, best_gi = thr, -1
            row = iou_mat[di]
            for gi in range(len(gts)):
                if used[gi]:
                    continue
                if row[gi] >= best_iou:
                    best_iou = row[gi]
                    best_gi = gi
            if best_gi >= 0:
                tp[di, ti] = True
                used[best_gi] = True
    return tp


def match_for_map(dets, gts, niou):
    return match_for_thresholds(dets, gts, IOU_THRESHOLDS[:niou])


def update_confusion_matrix(cm, dets, gts, num_classes, conf_thres=CONFMAT_CONF):
    """Class-agnostic matching at CONFMAT_IOU; record class confusion."""
    hi = [d for d in dets if d["score"] >= conf_thres]
    hi.sort(key=lambda d: -d["score"])
    gt_used = [False] * len(gts)
    det_matched = [False] * len(hi)
    for di, d in enumerate(hi):
        best_iou, best_gi = CONFMAT_IOU, -1
        for gi, (_, gp) in enumerate(gts):
            if gt_used[gi]:
                continue
            iou = poly_iou(d["poly"], gp)
            if iou >= best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0:
            gc = gts[best_gi][0]
            cm[gc, d["cls"]] += 1
            gt_used[best_gi] = True
            det_matched[di] = True
    # Predictions with no IoU match → background row
    for di, d in enumerate(hi):
        if not det_matched[di]:
            cm[num_classes, d["cls"]] += 1
    # GTs with no matched prediction → background column
    for gi, (gc, _) in enumerate(gts):
        if not gt_used[gi]:
            cm[gc, num_classes] += 1


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def use_pre_quantized_output(model_path):
    """Bypass final output Q/DQ when it collapses mixed-range YOLO outputs."""
    import onnx

    model_path = Path(model_path)
    model = onnx.load(model_path)
    replaced = False
    for graph_output in model.graph.output:
        output_name = graph_output.name
        dq_node = next(
            (node for node in model.graph.node
             if node.op_type == "DequantizeLinear" and output_name in node.output),
            None,
        )
        if dq_node is None:
            continue
        q_output = dq_node.input[0]
        q_node = next(
            (node for node in model.graph.node
             if node.op_type == "QuantizeLinear" and q_output in node.output),
            None,
        )
        if q_node is None:
            continue
        graph_output.name = q_node.input[0]
        replaced = True

    if not replaced:
        return str(model_path), False

    tmp_dir = Path(tempfile.gettempdir()) / "yolo_obb_int8_eval"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    patched_path = tmp_dir / f"{model_path.stem}-pre-output.onnx"
    onnx.save(model, patched_path)
    return str(patched_path), True


def summarize_high_conf_dets(dets, classes):
    counts = {}
    for d in dets:
        cname = classes[d["cls"]] if d["cls"] < len(classes) else str(d["cls"])
        counts[cname] = counts.get(cname, 0) + 1
    return "; ".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def raw_class_score_max(raw, num_classes):
    preds = np.squeeze(raw)
    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T
    expected = 4 + num_classes + 1
    if preds.ndim != 2 or preds.shape[1] < expected:
        return 0.0
    return float(np.nanmax(preds[:, 4:4 + num_classes]))


def evaluate(model_path, images_dir, labels_dir, providers, classes,
             ignore_empty_labels=False, confmat_conf=CONFMAT_CONF):
    session_model_path, patched_output = use_pre_quantized_output(model_path)
    if patched_output:
        print("  [INFO] Using pre-quantized output tensor for INT8 evaluation.")
    sess = ort.InferenceSession(session_model_path, providers=providers)
    inp = sess.get_inputs()[0]
    input_size = int(inp.shape[2])
    in_name = inp.name

    nc = len(classes)
    niou = len(IOU_THRESHOLDS)

    img_paths = sorted(
        p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    stats_tp = [[] for _ in range(nc)]      # list of (niou,) bool arrays
    stats_conf = [[] for _ in range(nc)]
    stats_tp25 = [[] for _ in range(nc)]
    stats_conf25 = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, dtype=np.int64)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    empty_label_detections = []

    inf_times = []
    evaluated_images = 0
    max_raw_class_score = 0.0
    t_total = time.perf_counter()
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        gts = load_gt(str(label_path), w, h)
        if ignore_empty_labels and not gts:
            if (i + 1) % 10 == 0 or i == len(img_paths) - 1:
                print(f"  [{i + 1:>4}/{len(img_paths)}] {img_path.name} (skipped empty label)")
            continue

        blob, meta = preprocess(img, input_size)

        t0 = time.perf_counter()
        outputs = sess.run(None, {in_name: blob})
        inf_times.append(time.perf_counter() - t0)
        evaluated_images += 1
        max_raw_class_score = max(
            max_raw_class_score, raw_class_score_max(outputs[0], nc)
        )

        dets = postprocess(
            outputs[0], input_size, w, h, nc, CONF_THRESHOLD_INFER, meta
        )
        dets.sort(key=lambda d: -d["score"])

        high_conf_empty = [d for d in dets if d["score"] >= confmat_conf]
        if not gts and high_conf_empty:
            empty_label_detections.append({
                "image": img_path.name,
                "count": len(high_conf_empty),
                "max_conf": max(d["score"] for d in high_conf_empty),
                "classes": summarize_high_conf_dets(high_conf_empty, classes),
            })

        for gc, _ in gts:
            if 0 <= gc < nc:
                n_gt[gc] += 1

        tp_arr = match_for_map(dets, gts, niou)
        tp25_arr = match_for_thresholds(dets, gts, [PERMISSIVE_IOU])
        for di, d in enumerate(dets):
            stats_tp[d["cls"]].append(tp_arr[di])
            stats_conf[d["cls"]].append(d["score"])
            stats_tp25[d["cls"]].append(tp25_arr[di, 0])
            stats_conf25[d["cls"]].append(d["score"])

        update_confusion_matrix(cm, dets, gts, nc, confmat_conf)

        if (i + 1) % 10 == 0 or i == len(img_paths) - 1:
            print(f"  [{i + 1:>4}/{len(img_paths)}] {img_path.name}")

    wall = time.perf_counter() - t_total

    # Aggregate AP per class per IoU threshold
    ap = np.zeros((nc, niou), dtype=np.float64)
    p_best = np.zeros(nc)
    r_best = np.zeros(nc)
    f1_best = np.zeros(nc)
    thr_best = np.zeros(nc)
    ap25 = np.zeros(nc, dtype=np.float64)
    p25_best = np.zeros(nc)
    r25_best = np.zeros(nc)
    f1_25_best = np.zeros(nc)
    thr25_best = np.zeros(nc)

    for c in range(nc):
        if not stats_tp[c] or n_gt[c] == 0:
            continue
        tp = np.array(stats_tp[c], dtype=bool)
        conf = np.array(stats_conf[c], dtype=np.float64)
        order = np.argsort(-conf)
        tp = tp[order]
        conf = conf[order]
        fp = ~tp
        tp_cum = tp.cumsum(0)
        fp_cum = fp.cumsum(0)
        recall = tp_cum / max(n_gt[c], 1)
        precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
        for ti in range(niou):
            ap[c, ti] = compute_ap(recall[:, ti], precision[:, ti])

        # P / R / F1 at IoU=0.5, picking conf that maximises F1
        rec_50 = recall[:, 0]
        pre_50 = precision[:, 0]
        f1_50 = 2 * pre_50 * rec_50 / np.maximum(pre_50 + rec_50, 1e-16)
        idx = int(np.argmax(f1_50))
        p_best[c] = pre_50[idx]
        r_best[c] = rec_50[idx]
        f1_best[c] = f1_50[idx]
        thr_best[c] = conf[idx]

        if stats_tp25[c]:
            tp25 = np.array(stats_tp25[c], dtype=bool)
            conf25 = np.array(stats_conf25[c], dtype=np.float64)
            order25 = np.argsort(-conf25)
            tp25 = tp25[order25]
            conf25 = conf25[order25]
            fp25 = ~tp25
            tp25_cum = tp25.cumsum(0)
            fp25_cum = fp25.cumsum(0)
            rec25 = tp25_cum / max(n_gt[c], 1)
            pre25 = tp25_cum / np.maximum(tp25_cum + fp25_cum, 1)
            ap25[c] = compute_ap(rec25, pre25)
            f1_25 = 2 * pre25 * rec25 / np.maximum(pre25 + rec25, 1e-16)
            idx25 = int(np.argmax(f1_25))
            p25_best[c] = pre25[idx25]
            r25_best[c] = rec25[idx25]
            f1_25_best[c] = f1_25[idx25]
            thr25_best[c] = conf25[idx25]

    return {
        "classes": classes,
        "n_gt": n_gt,
        "ap": ap,                   # (nc, niou)
        "map50": float(ap[:, 0].mean()),
        "map5095": float(ap.mean()),
        "precision": p_best,
        "recall": r_best,
        "f1": f1_best,
        "conf_thr": thr_best,
        "ap25": ap25,
        "map25": float(ap25.mean()),
        "precision25": p25_best,
        "recall25": r25_best,
        "f1_25": f1_25_best,
        "conf_thr25": thr25_best,
        "confusion_matrix": cm,
        "inference_ms_mean": 1000.0 * float(np.mean(inf_times)) if inf_times else 0.0,
        "inference_ms_p95": 1000.0 * float(np.percentile(inf_times, 95)) if inf_times else 0.0,
        "wall_seconds": wall,
        "input_size": input_size,
        "num_images": evaluated_images,
        "num_input_images": len(img_paths),
        "empty_label_detections": empty_label_detections,
        "confmat_conf": confmat_conf,
        "raw_class_score_max": max_raw_class_score,
        "patched_output": patched_output,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_per_class(name, result):
    classes = result["classes"]
    n_gt = result["n_gt"]
    ap = result["ap"]
    print(
        f"\n--- {name} per-class metrics "
        f"(standard IoU=0.5, permissive IoU={PERMISSIVE_IOU}) ---"
    )
    header = (
        f"{'class':<22}{'GT':>6}"
        f"{'P50':>9}{'R50':>9}{'F1-50':>9}{'AP50':>9}{'AP50-95':>10}"
        f"{'P25':>9}{'R25':>9}{'F1-25':>9}{'AP25':>9}{'conf25':>9}"
    )
    print(header)
    print("-" * len(header))
    for c, name_c in enumerate(classes):
        ap50 = ap[c, 0]
        ap_mean = ap[c].mean()
        print(
            f"{name_c:<22}{int(n_gt[c]):>6}"
            f"{result['precision'][c]:>9.4f}"
            f"{result['recall'][c]:>9.4f}"
            f"{result['f1'][c]:>9.4f}"
            f"{ap50:>9.4f}"
            f"{ap_mean:>10.4f}"
            f"{result['precision25'][c]:>9.4f}"
            f"{result['recall25'][c]:>9.4f}"
            f"{result['f1_25'][c]:>9.4f}"
            f"{result['ap25'][c]:>9.4f}"
            f"{result['conf_thr25'][c]:>9.4f}"
        )
    print("-" * len(header))
    print(
        f"{'mean':<22}{int(n_gt.sum()):>6}"
        f"{result['precision'].mean():>9.4f}"
        f"{result['recall'].mean():>9.4f}"
        f"{result['f1'].mean():>9.4f}"
        f"{result['map50']:>9.4f}"
        f"{result['map5095']:>10.4f}"
        f"{result['precision25'].mean():>9.4f}"
        f"{result['recall25'].mean():>9.4f}"
        f"{result['f1_25'].mean():>9.4f}"
        f"{result['map25']:>9.4f}"
    )
    print(
        f"\nInference: mean={result['inference_ms_mean']:.2f} ms  "
        f"p95={result['inference_ms_p95']:.2f} ms  "
        f"wall={result['wall_seconds']:.2f} s "
        f"({result['num_images']} imgs, input {result['input_size']}px)"
    )


def print_empty_label_warning(name, result):
    rows = result["empty_label_detections"]
    if not rows:
        return
    confmat_conf = result["confmat_conf"]
    total = sum(r["count"] for r in rows)
    print(
        f"\n[WARN] {name}: {len(rows)} images have no GT labels but produced "
        f"{total} predictions at conf>={confmat_conf}."
    )
    print("       These are counted as background false positives. Top examples:")
    for row in sorted(rows, key=lambda r: -r["max_conf"])[:8]:
        print(
            f"       {row['image']}: {row['count']} preds, "
            f"max_conf={row['max_conf']:.3f}, {row['classes']}"
        )


def print_int8_output_warning(name, result):
    max_score = result["raw_class_score_max"]
    if max_score > CONF_THRESHOLD_INFER:
        return
    print(
        f"\n[WARN] {name}: max raw class score is {max_score:.6f}. "
        "This INT8 ONNX produced no usable class-confidence outputs, so zero "
        "detections are expected. Check the quantization/export pipeline."
    )


def print_confusion_matrix(name, cm, classes, confmat_conf=CONFMAT_CONF):
    labels = list(classes) + ["background"]
    width = max(max(len(l) for l in labels), 10)
    print(f"\n--- {name} confusion matrix "
          f"(rows = GT, cols = prediction; IoU≥{CONFMAT_IOU}, conf≥{confmat_conf}) ---")
    header = " " * (width + 2) + "".join(f"{l:>{width + 2}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        line = f"{row_label:<{width + 2}}" + "".join(
            f"{int(cm[i, j]):>{width + 2}}" for j in range(len(labels))
        )
        print(line)


def save_metrics_csv(path, name, result):
    classes = result["classes"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", name])
        w.writerow([])
        w.writerow([
            "class", "GT",
            "precision50", "recall50", "F1-50", "AP50", "AP50-95", "best_conf50",
            "precision25", "recall25", "F1-25", "AP25", "best_conf25",
        ])
        for c, cname in enumerate(classes):
            w.writerow([
                cname,
                int(result["n_gt"][c]),
                f"{result['precision'][c]:.6f}",
                f"{result['recall'][c]:.6f}",
                f"{result['f1'][c]:.6f}",
                f"{result['ap'][c, 0]:.6f}",
                f"{result['ap'][c].mean():.6f}",
                f"{result['conf_thr'][c]:.6f}",
                f"{result['precision25'][c]:.6f}",
                f"{result['recall25'][c]:.6f}",
                f"{result['f1_25'][c]:.6f}",
                f"{result['ap25'][c]:.6f}",
                f"{result['conf_thr25'][c]:.6f}",
            ])
        w.writerow([
            "mean",
            int(result["n_gt"].sum()),
            f"{result['precision'].mean():.6f}",
            f"{result['recall'].mean():.6f}",
            f"{result['f1'].mean():.6f}",
            f"{result['map50']:.6f}",
            f"{result['map5095']:.6f}",
            "",
            f"{result['precision25'].mean():.6f}",
            f"{result['recall25'].mean():.6f}",
            f"{result['f1_25'].mean():.6f}",
            f"{result['map25']:.6f}",
            "",
        ])
        w.writerow([])
        w.writerow(["inference_ms_mean", f"{result['inference_ms_mean']:.4f}"])
        w.writerow(["inference_ms_p95", f"{result['inference_ms_p95']:.4f}"])
        w.writerow(["wall_seconds", f"{result['wall_seconds']:.4f}"])
        w.writerow(["num_images", result["num_images"]])
        w.writerow(["input_size", result["input_size"]])


def save_metrics_png(path, name, result):
    """Optional matplotlib table of per-class metrics."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping {path}: {exc})")
        return

    classes = result["classes"]
    columns = [
        "class", "P50", "R50", "F1-50", "AP50", "AP50-95",
        "P25", "R25", "F1-25", "AP25", "conf25",
    ]
    rows = []
    for c, cname in enumerate(classes):
        rows.append([
            cname,
            f"{result['precision'][c]:.4f}",
            f"{result['recall'][c]:.4f}",
            f"{result['f1'][c]:.4f}",
            f"{result['ap'][c, 0]:.4f}",
            f"{result['ap'][c].mean():.4f}",
            f"{result['precision25'][c]:.4f}",
            f"{result['recall25'][c]:.4f}",
            f"{result['f1_25'][c]:.4f}",
            f"{result['ap25'][c]:.4f}",
            f"{result['conf_thr25'][c]:.4f}",
        ])
    rows.append([
        "mean",
        f"{result['precision'].mean():.4f}",
        f"{result['recall'].mean():.4f}",
        f"{result['f1'].mean():.4f}",
        f"{result['map50']:.4f}",
        f"{result['map5095']:.4f}",
        f"{result['precision25'].mean():.4f}",
        f"{result['recall25'].mean():.4f}",
        f"{result['f1_25'].mean():.4f}",
        f"{result['map25']:.4f}",
        "",
    ])

    fig_h = 1.1 + 0.42 * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(15, fig_h))
    ax.axis("off")
    ax.set_title(f"{name} — per-class metrics", fontsize=16, fontweight="bold", pad=8)
    ax.text(
        0.5, 0.88,
        f"NMS_IOU={NMS_IOU:.2f} | standard IoU=0.50 | permissive IoU={PERMISSIVE_IOU:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#4B5563",
    )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        bbox=[0.0, 0.02, 1.0, 0.72],
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.45)

    n_rows = len(rows) + 1
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#1F4E79")
            cell.set_text_props(color="white", weight="bold")
        elif row == n_rows - 1:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F6F8FA")
        if col == 0 and row > 0:
            cell.set_text_props(ha="left")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.06)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_csv(path, cm, classes):
    labels = list(classes) + ["background"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ pred", *labels])
        for i, row_label in enumerate(labels):
            w.writerow([row_label, *[int(cm[i, j]) for j in range(len(labels))]])


def save_empty_label_report(path, result):
    rows = result["empty_label_detections"]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "count", "max_conf", "classes"])
        w.writeheader()
        for row in sorted(rows, key=lambda r: -r["max_conf"]):
            w.writerow({
                "image": row["image"],
                "count": row["count"],
                "max_conf": f"{row['max_conf']:.6f}",
                "classes": row["classes"],
            })


def save_confusion_png(path, cm, classes, title):
    """Optional matplotlib plot. Silently skip if matplotlib is not installed."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping {path}: {exc})")
        return

    labels = list(classes) + ["background"]
    norm = cm.astype(np.float64)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = norm / row_sums

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.1 * len(labels) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{title} (row-normalized)", pad=22)
    ax.text(
        0.5, 1.02,
        f"NMS_IOU={NMS_IOU:.2f} | CONFMAT_IOU={CONFMAT_IOU:.2f} | CONFMAT_CONF={CONFMAT_CONF:.2f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            pct = 100.0 * norm[i, j]
            text = f"{pct:.1f}%\n({int(cm[i, j])})" if cm[i, j] else "0.0%"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_comparison(summaries):
    if len(summaries) < 2:
        return
    print("\n=== Model comparison (mean over classes) ===")
    header = f"{'model':<22}{'P':>10}{'R':>10}{'F1':>10}{'mAP50':>10}{'mAP50-95':>12}{'ms/img':>10}"
    print(header)
    print("-" * len(header))
    for name, r in summaries.items():
        print(
            f"{name:<22}"
            f"{r['precision'].mean():>10.4f}"
            f"{r['recall'].mean():>10.4f}"
            f"{r['f1'].mean():>10.4f}"
            f"{r['map50']:>10.4f}"
            f"{r['map5095']:>12.4f}"
            f"{r['inference_ms_mean']:>10.2f}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=DEFAULT_VAL_IMAGES,
                    help="Directory of validation images.")
    ap.add_argument("--labels", default=DEFAULT_VAL_LABELS,
                    help="Directory of YOLO-OBB .txt label files.")
    ap.add_argument("--model", action="append", default=None,
                    metavar="NAME=PATH",
                    help="Model entry (repeatable). Defaults to the bundled "
                         "YOLOv8/YOLO26 INT8 OBB ONNX models.")
    ap.add_argument("--output-dir", default="benchmark_results/obb_int8_eval",
                    help="Where to write CSV / PNG outputs.")
    ap.add_argument("--providers", nargs="+",
                    default=["CPUExecutionProvider"],
                    help="ONNX Runtime execution providers.")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Class names in order (must match training).")
    ap.add_argument("--ignore-empty-labels", action="store_true",
                    help="Skip images with no GT boxes. Useful when empty label "
                         "files may actually be unlabeled positives.")
    ap.add_argument("--confmat-conf", type=float, default=CONFMAT_CONF,
                    help="Confidence threshold for the confusion matrix only. "
                         "AP/P/R/F1 are still computed from the full confidence sweep.")
    return ap.parse_args()


def parse_model_args(items):
    out = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--model expects NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def main():
    args = parse_args()
    models = parse_model_args(args.model) if args.model else dict(DEFAULT_MODELS)

    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    os.makedirs(args.output_dir, exist_ok=True)

    summaries = {}
    for name, mp in models.items():
        if not os.path.isfile(mp):
            print(f"[WARN] {name}: model not found at {mp} — skipping")
            continue
        print(f"\n=== Evaluating {name} ({mp}) ===")
        result = evaluate(
            mp, args.images, args.labels, args.providers, args.classes,
            ignore_empty_labels=args.ignore_empty_labels,
            confmat_conf=args.confmat_conf,
        )
        summaries[name] = result

        print_per_class(name, result)
        print_int8_output_warning(name, result)
        print_empty_label_warning(name, result)
        print_confusion_matrix(
            name, result["confusion_matrix"], args.classes, args.confmat_conf
        )

        slug = name.lower().replace("/", "_").replace(" ", "_")
        save_metrics_csv(
            os.path.join(args.output_dir, f"{slug}_metrics.csv"), name, result
        )
        save_metrics_png(
            os.path.join(args.output_dir, f"{slug}_metrics.png"), name, result
        )
        save_confusion_csv(
            os.path.join(args.output_dir, f"{slug}_confusion_matrix.csv"),
            result["confusion_matrix"],
            args.classes,
        )
        save_empty_label_report(
            os.path.join(args.output_dir, f"{slug}_empty_label_detections.csv"),
            result,
        )
        save_confusion_png(
            os.path.join(args.output_dir, f"{slug}_confusion_matrix.png"),
            result["confusion_matrix"],
            args.classes,
            f"{name} — confusion matrix",
        )

    if not summaries:
        raise SystemExit("No models evaluated.")

    print_comparison(summaries)
    print(f"\nArtifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
