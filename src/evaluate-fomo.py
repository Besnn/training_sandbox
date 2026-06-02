#!/usr/bin/env python3
"""Evaluate FOMO ONNX models with a simple, distance-only metric.

FOMO is a centroid detector: the model outputs a (num_classes+1) x H x W logit
map where channel 0 is background. This script keeps detection deliberately
simple — ONE detection per active grid cell, placed at that cell's NORMALIZED
centre ((x+0.5)/W, (y+0.5)/H) in [0,1] image space, with no clustering and no
grid-coordinate mapping. Quality is measured by matching predicted centroids to
ground-truth centroids by plain L2 distance in normalized space; that distance
plays the role IoU plays for box detectors, and a distance sweep gives an
mAP-like aggregate.

The reporting is identical to evaluate-fomo.py: per-class P/R/F1, AP@d and an
AP distance sweep, a row-normalized confusion matrix, CSV/PNG artifacts, and a
multi-model comparison table. Only the detection (per-cell, no clustering) and
the distance units (normalized, not grid cells) differ.

YOLO-centroid label format (normalized, one centroid per line):
    class_id x_center y_center [w h ...]   # only the first three are used

Usage:
    python3 evaluate-fomo-simple.py
    python3 evaluate-fomo-simple.py \
        --images <dir> --labels <dir> \
        --model "FOMO-FP32=models/fomo_fp32.onnx" \
        --model "FOMO-INT8=models/fomo_int8.onnx"
"""

import argparse
import csv
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy.optimize import linear_sum_assignment

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
DEFAULT_MODELS = {
    "FOMO-FP32": "models/fomo-480-onnx/fomo-480.onnx",
    "FOMO-INT8": "models/fomo-480-onnx/fomo-480-int8.onnx",
}
DEFAULT_VAL_IMAGES = str(SCRIPT_DIR / "datasets/yolo_pl_test/images")
DEFAULT_VAL_LABELS = str(SCRIPT_DIR / "datasets/yolo_pl_test/labels")
DEFAULT_OUTPUT_DIR = str(SCRIPT_DIR / "benchmark_results/precision-recall/")

# All distances below are in NORMALIZED image units (fraction of width/height),
# so they are independent of the model's grid resolution.
#
# Distance sweep for the mAP-like aggregate (10 thresholds, COCO-sized).
DISTANCE_THRESHOLDS = np.linspace(0.025, 0.125, 10)

# Primary distance for the headline P/R/F1 and AP@d numbers.
#  sqrt(0.5^2 + 0.5^2) * 1/60 = 0.7078 * 1/60 = 0.012
PRIMARY_DISTANCE = 0.025
# Loose confidence kept while sweeping the P/R curve.
CONF_THRESHOLD_INFER = 0.05
# Distance and confidence used for the confusion matrix only.
CONFMAT_DIST = 0.1
CONFMAT_CONF = 0.25

EVAL_THREADS = 4


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_gt(label_path, grid_w=None, grid_h=None, raw_w=None, raw_h=None):
    """Return list of (class_id, (x, y)) with x, y NORMALIZED to [0, 1].

    Supports two YOLO label layouts, both normalized:
      - centroid: class x_center y_center [w h ...]  -> uses fields 1,2
      - polygon:  class x1 y1 x2 y2 x3 y3 x4 y4      -> centroid = mean of corners
    The dataset here ships polygons, so taking fields 1,2 directly would read a
    CORNER, not the centre — a size-dependent offset that wrecks distance matching.

    grid_w/grid_h/raw_w/raw_h are accepted for compatibility with the FOMO
    inspection scripts. They are deliberately ignored because this evaluator's
    postprocess and metrics use normalized image coordinates throughout.
    """
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            p = line.split()
            if len(p) < 3:
                continue
            cls = int(p[0])
            if len(p) == 9:  # 4 corner points
                x = sum(float(p[i]) for i in (1, 3, 5, 7)) / 4.0
                y = sum(float(p[i]) for i in (2, 4, 6, 8)) / 4.0
            else:
                x, y = float(p[1]), float(p[2])
            out.append((cls, (x, y)))
    return out


# --------------------------------------------------------------------------- #
# FOMO inference
# --------------------------------------------------------------------------- #
def preprocess(image, input_size):
    """Resize to (input_size, input_size), BGR->RGB, scale to [0,1], CHW, batch."""
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)
    return np.expand_dims(blob, 0)


def softmax(logits, axis):
    e = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def postprocess(raw, num_classes, conf_thres):
    """One detection per active cell at its normalized centre.

    No clustering, no grid mapping — a cell at (col, row) in an H x W grid sits
    at ((col+0.5)/W, (row+0.5)/H), the same [0,1] space ground truth uses.
    Returns dicts shaped like the full pipeline: {"cell": (x, y), "cls", "score"}.
    """
    arr = np.asarray(raw, dtype=np.float32)  # cast first: avoids int8 exp overflow
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected FOMO output shape: {arr.shape}")

    c = arr.shape[0]
    if c != num_classes + 1:
        if arr.shape[-1] == num_classes + 1:  # recover channels-last exports
            arr = arr.transpose(2, 0, 1)
        else:
            raise ValueError(
                f"FOMO output has {arr.shape[0]} channels, expected {num_classes + 1}"
            )

    probs = softmax(arr, axis=0)
    pred_map = probs.argmax(axis=0)
    score_map = probs.max(axis=0)
    h, w = score_map.shape

    dets = []
    # FOMO channel 0 is background. A cell is a detection only when the best
    # channel is foreground; otherwise weak foreground probability in a
    # background-winning cell becomes a false detection and corrupts P/R.
    active = (pred_map != 0) & (score_map >= conf_thres)
    for y, x in zip(*np.where(active)):
        dets.append({
            "cell": ((int(x) + 0.5) / w, (int(y) + 0.5) / h),
            "cls": int(pred_map[y, x] - 1),
            "score": float(score_map[y, x]),
        })
    return dets


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def compute_ap(recall, precision):
    """101-point interpolated AP (COCO-style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def centroid_distance_matrix(dets, gts):
    """Pairwise normalized Euclidean distance between dets and gts."""
    if not dets or not gts:
        return np.zeros((len(dets), len(gts)), dtype=np.float32)
    d_xy = np.array([d["cell"] for d in dets], dtype=np.float32)
    g_xy = np.array([g[1] for g in gts], dtype=np.float32)
    diff = d_xy[:, None, :] - g_xy[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def match_for_map(dets, gts, ndist):
    """(N_det, ndist) bool TP array using confidence-ordered AP matching."""
    tp = np.zeros((len(dets), ndist), dtype=bool)
    if not dets or not gts:
        return tp
    dist_mat = centroid_distance_matrix(dets, gts)
    same_class = np.array(
        [[d["cls"] == gc for gc, _ in gts] for d in dets], dtype=bool
    )
    order = np.argsort([-d["score"] for d in dets])

    for ti, thr in enumerate(DISTANCE_THRESHOLDS):
        used_gts = np.zeros(len(gts), dtype=bool)
        # AP is defined by walking detections in confidence order. A global
        # Hungarian solve can give a lower-confidence detection the GT and turn
        # the actual high-confidence hit into an FP.
        for di in order:
            candidates = np.where(
                same_class[di] & (~used_gts) & (dist_mat[di] <= thr)
            )[0]
            if candidates.size == 0:
                continue
            gi = int(candidates[np.argmin(dist_mat[di, candidates])])
            used_gts[gi] = True
            tp[di, ti] = True
    return tp


def update_confusion_matrix(cm, dets, gts, num_classes,
                            conf_thres=CONFMAT_CONF, dist_thres=CONFMAT_DIST):
    """Two-pass Hungarian: same-class TPs first, then cross-class confusion."""
    hi = [d for d in dets if d["score"] >= conf_thres]
    if not hi and not gts:
        return

    gt_used = np.zeros(len(gts), dtype=bool)
    det_matched = np.zeros(len(hi), dtype=bool)

    if hi and gts:
        d_xy = np.array([d["cell"] for d in hi], dtype=np.float32)
        g_xy = np.array([g[1] for g in gts], dtype=np.float32)
        diff = d_xy[:, None, :] - g_xy[None, :, :]
        dist_mat = np.sqrt((diff * diff).sum(-1))

        same_class = np.array(
            [[d["cls"] == gc for gc, _ in gts] for d in hi], dtype=bool
        )

        # Pass 1: optimal same-class matching (TPs)
        cost1 = np.where(same_class & (dist_mat <= dist_thres), dist_mat, 1e9)
        r1, c1 = linear_sum_assignment(cost1)
        for di, gi in zip(r1, c1):
            if same_class[di, gi] and dist_mat[di, gi] <= dist_thres:
                cm[gts[gi][0], hi[di]["cls"]] += 1
                gt_used[gi] = True
                det_matched[di] = True

        # Pass 2: optimal cross-class matching among leftovers (class confusion)
        rem_dets = np.where(~det_matched)[0]
        rem_gts = np.where(~gt_used)[0]
        if rem_dets.size and rem_gts.size:
            sub = dist_mat[np.ix_(rem_dets, rem_gts)]
            cost2 = np.where(sub <= dist_thres, sub, 1e9)
            r2, c2 = linear_sum_assignment(cost2)
            for ri, ci in zip(r2, c2):
                di, gi = int(rem_dets[ri]), int(rem_gts[ci])
                if dist_mat[di, gi] <= dist_thres:
                    cm[gts[gi][0], hi[di]["cls"]] += 1
                    gt_used[gi] = True
                    det_matched[di] = True

    # Pass 3: residual FPs and FNs
    for di, d in enumerate(hi):
        if not det_matched[di]:
            cm[num_classes, d["cls"]] += 1  # fired with no nearby GT
    for gi, (gc, _) in enumerate(gts):
        if not gt_used[gi]:
            cm[gc, num_classes] += 1  # GT missed entirely


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def summarize_high_conf_dets(dets, classes):
    counts = {}
    for d in dets:
        cname = classes[d["cls"]] if d["cls"] < len(classes) else str(d["cls"])
        counts[cname] = counts.get(cname, 0) + 1
    return "; ".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def make_session_options(num_threads=EVAL_THREADS):
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    return options


def configure_providers(providers, num_threads=EVAL_THREADS):
    configured = []
    for provider in providers:
        if provider == "XnnpackExecutionProvider":
            configured.append((provider, {"intra_op_num_threads": str(num_threads)}))
        else:
            configured.append(provider)
    return configured


def infer_grid_size(session, input_size, input_name):
    """Run one dummy inference and return the output grid as (width, height)."""
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    out = session.run(None, {input_name: dummy})[0]
    arr = np.asarray(out)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected FOMO output shape while inferring grid: {arr.shape}")

    # Outputs are normally CHW, but some exports are HWC. Detect the channel
    # axis from the class count so callers get the real spatial grid either way.
    expected_channels = len(DEFAULT_CLASSES) + 1
    if arr.shape[0] == expected_channels:
        return int(arr.shape[2]), int(arr.shape[1])
    if arr.shape[-1] == expected_channels:
        return int(arr.shape[1]), int(arr.shape[0])
    return int(arr.shape[-1]), int(arr.shape[-2])


def evaluate(model_path, images_dir, labels_dir, providers, classes,
             ignore_empty_labels=False, confmat_conf=CONFMAT_CONF,
             confmat_dist=CONFMAT_DIST):
    sess = ort.InferenceSession(
        model_path,
        sess_options=make_session_options(EVAL_THREADS),
        providers=configure_providers(providers, EVAL_THREADS),
    )
    inp = sess.get_inputs()[0]
    in_name = inp.name
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 480

    nc = len(classes)
    ndist = len(DISTANCE_THRESHOLDS)

    img_paths = sorted(
        p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMG_EXTS
    )
    if not img_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    stats_tp = [[] for _ in range(nc)]
    stats_conf = [[] for _ in range(nc)]
    n_gt = np.zeros(nc, dtype=np.int64)
    cm = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    empty_label_detections = []

    inf_times = []
    evaluated_images = 0
    grid_w = grid_h = 0
    t_total = time.perf_counter()

    for completed, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gts = [g for g in load_gt(str(Path(labels_dir) / (img_path.stem + ".txt")))
               if 0 <= g[0] < nc]
        if ignore_empty_labels and not gts:
            if completed % 10 == 0 or completed == len(img_paths):
                print(f"  [{completed:>4}/{len(img_paths)}] {img_path.name} (skipped empty label)")
            continue

        blob = preprocess(img, input_size)
        t0 = time.perf_counter()
        out = sess.run(None, {in_name: blob})[0]
        inf_times.append(time.perf_counter() - t0)
        evaluated_images += 1

        if not grid_w:
            shp = np.asarray(out).shape
            grid_h, grid_w = (shp[-2], shp[-1])

        dets = postprocess(out, nc, CONF_THRESHOLD_INFER)
        dets.sort(key=lambda d: -d["score"])

        high_conf = [d for d in dets if d["score"] >= confmat_conf]
        if not gts and high_conf:
            empty_label_detections.append({
                "image": img_path.name,
                "count": len(high_conf),
                "max_conf": max(d["score"] for d in high_conf),
                "classes": summarize_high_conf_dets(high_conf, classes),
            })

        for gc, _ in gts:
            n_gt[gc] += 1

        tp_arr = match_for_map(dets, gts, ndist)
        for di, d in enumerate(dets):
            stats_tp[d["cls"]].append(tp_arr[di])
            stats_conf[d["cls"]].append(d["score"])

        update_confusion_matrix(cm, dets, gts, nc, confmat_conf, confmat_dist)

        if completed % 10 == 0 or completed == len(img_paths):
            print(f"  [{completed:>4}/{len(img_paths)}] {img_path.name}")

    wall = time.perf_counter() - t_total

    ap = np.zeros((nc, ndist), dtype=np.float64)
    p_best = np.zeros(nc)
    r_best = np.zeros(nc)
    f1_best = np.zeros(nc)
    thr_best = np.zeros(nc)
    primary_ti = int(np.argmin(np.abs(DISTANCE_THRESHOLDS - PRIMARY_DISTANCE)))

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
        for ti in range(ndist):
            ap[c, ti] = compute_ap(recall[:, ti], precision[:, ti])

        rec_p = recall[:, primary_ti]
        pre_p = precision[:, primary_ti]
        f1_p = 2 * pre_p * rec_p / np.maximum(pre_p + rec_p, 1e-16)
        idx = int(np.argmax(f1_p))
        p_best[c] = pre_p[idx]
        r_best[c] = rec_p[idx]
        f1_best[c] = f1_p[idx]
        thr_best[c] = conf[idx]

    return {
        "classes": classes,
        "n_gt": n_gt,
        "ap": ap,
        "ap_primary": float(ap[:, primary_ti].mean()),
        "ap_mean": float(ap.mean()),
        "distance_thresholds": DISTANCE_THRESHOLDS,
        "primary_ti": primary_ti,
        "primary_distance": float(DISTANCE_THRESHOLDS[primary_ti]),
        "precision": p_best,
        "recall": r_best,
        "f1": f1_best,
        "conf_thr": thr_best,
        "confusion_matrix": cm,
        "inference_ms_mean": 1000.0 * float(np.mean(inf_times)) if inf_times else 0.0,
        "inference_ms_p95": 1000.0 * float(np.percentile(inf_times, 95)) if inf_times else 0.0,
        "wall_seconds": wall,
        "input_size": input_size,
        "grid_size": (grid_w, grid_h),
        "num_images": evaluated_images,
        "num_input_images": len(img_paths),
        "empty_label_detections": empty_label_detections,
        "confmat_conf": confmat_conf,
        "confmat_dist": confmat_dist,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_per_class(name, result):
    classes = result["classes"]
    n_gt = result["n_gt"]
    ap = result["ap"]
    primary = result["primary_distance"]
    print(f"\n--- {name} per-class metrics (distance={primary:.3f} norm for P/R/F1) ---")
    header = (
        f"{'class':<22}{'GT':>6}{'P':>10}{'R':>10}{'F1':>10}"
        f"{'AP@d=' + format(primary, '.3f'):>12}{'AP@d=sweep':>12}{'conf*':>10}"
    )
    print(header)
    print("-" * len(header))
    primary_ti = result["primary_ti"]
    for c, name_c in enumerate(classes):
        print(
            f"{name_c:<22}{int(n_gt[c]):>6}"
            f"{result['precision'][c]:>10.4f}"
            f"{result['recall'][c]:>10.4f}"
            f"{result['f1'][c]:>10.4f}"
            f"{ap[c, primary_ti]:>12.4f}"
            f"{ap[c].mean():>12.4f}"
            f"{result['conf_thr'][c]:>10.4f}"
        )
    print("-" * len(header))
    print(
        f"{'mean':<22}{int(n_gt.sum()):>6}"
        f"{result['precision'].mean():>10.4f}"
        f"{result['recall'].mean():>10.4f}"
        f"{result['f1'].mean():>10.4f}"
        f"{result['ap_primary']:>12.4f}"
        f"{result['ap_mean']:>12.4f}"
    )
    gw, gh = result["grid_size"]
    print(
        f"\nInference: mean={result['inference_ms_mean']:.2f} ms  "
        f"p95={result['inference_ms_p95']:.2f} ms  "
        f"wall={result['wall_seconds']:.2f} s "
        f"({result['num_images']} imgs, input {result['input_size']}px, "
        f"grid {gw}x{gh})"
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


def row_normalize_confusion_matrix(cm):
    norm = cm.astype(np.float64)
    row_sums = norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return norm / row_sums


def print_confusion_matrix(name, cm, classes, confmat_conf, confmat_dist):
    labels = list(classes) + ["background"]
    norm = row_normalize_confusion_matrix(cm)
    width = max(max(len(l) for l in labels), 10)
    print(
        f"\n--- {name} confusion matrix "
        f"(row-normalized; dist≤{confmat_dist}, conf≥{confmat_conf}) ---"
    )
    header = " " * (width + 2) + "".join(f"{l:>{width + 2}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        line = f"{row_label:<{width + 2}}" + "".join(
            f"{norm[i, j]:>{width + 1}.3f} " for j in range(len(labels))
        )
        print(line)


def save_metrics_csv(path, name, result):
    classes = result["classes"]
    primary = result["primary_distance"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", name])
        w.writerow([])
        w.writerow([
            "class", "GT", "precision", "recall", "F1",
            f"AP@d={primary:.3f}", "AP@d=sweep_mean",
        ])
        for c, cname in enumerate(classes):
            w.writerow([
                cname,
                int(result["n_gt"][c]),
                f"{result['precision'][c]:.6f}",
                f"{result['recall'][c]:.6f}",
                f"{result['f1'][c]:.6f}",
                f"{result['ap'][c, result['primary_ti']]:.6f}",
                f"{result['ap'][c].mean():.6f}",
            ])
        w.writerow([
            "mean",
            int(result["n_gt"].sum()),
            f"{result['precision'].mean():.6f}",
            f"{result['recall'].mean():.6f}",
            f"{result['f1'].mean():.6f}",
            f"{result['ap_primary']:.6f}",
            f"{result['ap_mean']:.6f}",
        ])
        w.writerow([])
        w.writerow(["distance_thresholds", ",".join(f"{t:.3f}" for t in result["distance_thresholds"])])
        w.writerow(["grid_size", f"{result['grid_size'][0]}x{result['grid_size'][1]}"])
        w.writerow(["inference_ms_mean", f"{result['inference_ms_mean']:.4f}"])
        w.writerow(["inference_ms_p95", f"{result['inference_ms_p95']:.4f}"])
        w.writerow(["wall_seconds", f"{result['wall_seconds']:.4f}"])
        w.writerow(["num_images", result["num_images"]])
        w.writerow(["input_size", result["input_size"]])


def save_table_png_cv2(path, title, sections):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    title_scale = 0.8
    thickness = 1
    margin = 24
    line_h = 30
    section_gap = 18
    char_w = 9

    section_layouts = []
    max_width = 0
    total_height = margin + 36
    for subtitle, columns, rows in sections:
        col_widths = []
        for col_i, column in enumerate(columns):
            values = [str(column), *[str(row[col_i]) for row in rows]]
            col_widths.append(max(88, max(len(value) for value in values) * char_w + 18))
        table_width = sum(col_widths)
        section_height = 24 + line_h * (len(rows) + 1) + section_gap
        section_layouts.append((subtitle, columns, rows, col_widths, table_width, section_height))
        max_width = max(max_width, table_width)
        total_height += section_height

    width = max(900, max_width + 2 * margin)
    img = np.full((total_height + margin, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, title, (margin, margin + 10), font, title_scale, (23, 32, 51), 2, cv2.LINE_AA)
    y = margin + 48
    for subtitle, columns, rows, col_widths, table_width, _ in section_layouts:
        x = margin
        cv2.putText(img, subtitle, (x, y), font, scale, (23, 32, 51), 2, cv2.LINE_AA)
        y += 12
        cv2.rectangle(img, (x, y), (x + table_width, y + line_h), (31, 78, 121), -1)
        cx = x
        for col_i, column in enumerate(columns):
            cv2.putText(img, str(column), (cx + 8, y + 20), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            cx += col_widths[col_i]
        y += line_h
        for row_i, row in enumerate(rows):
            fill = (246, 248, 250) if row_i % 2 == 0 else (255, 255, 255)
            cv2.rectangle(img, (x, y), (x + table_width, y + line_h), fill, -1)
            cx = x
            for col_i, value in enumerate(row):
                cv2.putText(img, str(value), (cx + 8, y + 20), font, scale, (30, 41, 59), thickness, cv2.LINE_AA)
                cx += col_widths[col_i]
            y += line_h
        y += section_gap
    cv2.imwrite(path, img)


def save_metrics_png(path, name, result):
    classes = result["classes"]
    primary = result["primary_distance"]
    columns = [
        "class", "precision", "recall", "F1",
        f"AP@d={primary:.3f}", "AP@d=sweep",
    ]
    rows = []
    for c, cname in enumerate(classes):
        rows.append([
            cname,
            f"{result['precision'][c]:.4f}",
            f"{result['recall'][c]:.4f}",
            f"{result['f1'][c]:.4f}",
            f"{result['ap'][c, result['primary_ti']]:.4f}",
            f"{result['ap'][c].mean():.4f}",
        ])
    rows.append([
        "mean",
        f"{result['precision'].mean():.4f}",
        f"{result['recall'].mean():.4f}",
        f"{result['f1'].mean():.4f}",
        f"{result['ap_primary']:.4f}",
        f"{result['ap_mean']:.4f}",
    ])

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, using OpenCV fallback for {path}: {exc})")
        save_table_png_cv2(path, f"{name} - per-class metrics", [("Metrics", columns, rows)])
        return

    fig_h = 1.1 + 0.42 * (len(rows) + 1)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    ax.set_title(f"{name} — per-class metrics", fontsize=16, fontweight="bold", pad=8)
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        bbox=[0.0, 0.02, 1.0, 0.78],
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

    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.06)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_csv(path, cm, classes):
    labels = list(classes) + ["background"]
    norm = row_normalize_confusion_matrix(cm)
    row_sums = cm.sum(axis=1)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ pred (row-normalized)", *labels, "support"])
        for i, row_label in enumerate(labels):
            w.writerow([
                row_label,
                *[f"{norm[i, j]:.6f}" for j in range(len(labels))],
                int(row_sums[i]),
            ])


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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping {path}: {exc})")
        return

    labels = list(classes) + ["background"]
    norm = row_normalize_confusion_matrix(cm)

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.1 * len(labels) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"{title} (row-normalized)")
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
    header = (
        f"{'model':<22}{'P':>10}{'R':>10}{'F1':>10}"
        f"{'AP@d=primary':>14}{'AP@d=sweep':>12}{'ms/img':>10}"
    )
    print(header)
    print("-" * len(header))
    for name, r in summaries.items():
        print(
            f"{name:<22}"
            f"{r['precision'].mean():>10.4f}"
            f"{r['recall'].mean():>10.4f}"
            f"{r['f1'].mean():>10.4f}"
            f"{r['ap_primary']:>14.4f}"
            f"{r['ap_mean']:>12.4f}"
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
                    help="Directory of YOLO-centroid .txt label files.")
    ap.add_argument("--model", action="append", default=None, metavar="NAME=PATH",
                    help="Model entry (repeatable). Defaults to the bundled FOMO models.")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help="Root directory for CSV / PNG outputs. Each model writes "
                         "to its own subfolder named after the model.")
    ap.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                    help="ONNX Runtime execution providers.")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Class names in order (must match training).")
    ap.add_argument("--ignore-empty-labels", action="store_true",
                    help="Skip images with no GT centroids.")
    ap.add_argument("--confmat-conf", type=float, default=CONFMAT_CONF,
                    help="Confidence threshold for the confusion matrix only.")
    ap.add_argument("--confmat-dist", type=float, default=CONFMAT_DIST,
                    help="Normalized distance threshold for confusion matrix matching.")
    return ap.parse_args()


def parse_model_args(items):
    out = {}
    for raw in items:
        if "=" not in raw:
            raise SystemExit(f"--model expects NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def resolve_model_path(path):
    model_path = Path(path)
    if model_path.is_absolute() or model_path.is_file():
        return str(model_path)
    script_relative = SCRIPT_DIR / model_path
    return str(script_relative if script_relative.is_file() else model_path)


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
        model_path = resolve_model_path(mp)
        if not os.path.isfile(model_path):
            print(f"[WARN] {name}: model not found at {mp} — skipping")
            continue
        print(f"\n=== Evaluating {name} ({model_path}) ===")
        result = evaluate(
            model_path, args.images, args.labels, args.providers, args.classes,
            ignore_empty_labels=args.ignore_empty_labels,
            confmat_conf=args.confmat_conf,
            confmat_dist=args.confmat_dist,
        )
        summaries[name] = result

        print_per_class(name, result)
        print_empty_label_warning(name, result)
        print_confusion_matrix(
            name, result["confusion_matrix"], args.classes,
            args.confmat_conf, args.confmat_dist,
        )

        slug = name.lower().replace("/", "_").replace(" ", "_")
        model_output_dir = os.path.join(args.output_dir, slug)
        os.makedirs(model_output_dir, exist_ok=True)
        save_metrics_csv(os.path.join(model_output_dir, "metrics.csv"), name, result)
        save_metrics_png(os.path.join(model_output_dir, "metrics.png"), name, result)
        save_confusion_csv(
            os.path.join(model_output_dir, "confusion_matrix.csv"),
            result["confusion_matrix"], args.classes,
        )
        save_empty_label_report(
            os.path.join(model_output_dir, "empty_label_detections.csv"), result,
        )
        save_confusion_png(
            os.path.join(model_output_dir, "confusion_matrix.png"),
            result["confusion_matrix"], args.classes,
            f"{name} — confusion matrix",
        )
        print(f"[OK] {name}: artifacts written to {model_output_dir}")

    if not summaries:
        raise SystemExit("No models evaluated.")

    print_comparison(summaries)
    print(f"\nArtifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
