#!/usr/bin/env python3
"""Evaluate FOMO (Faster Objects, More Objects) ONNX models.

FOMO is a centroid detector: the model outputs a (num_classes+1) x grid_h x
grid_w logit map where channel 0 is background. There are no bounding boxes,
so this script measures detection quality by matching predicted centroids
against ground-truth centroids in grid coordinates. Distance (in grid cells)
plays the role IoU plays for bounding-box detectors — a distance sweep gives
an mAP-like aggregate analogous to mAP@[0.5:0.95].

YOLO-centroid label format (one centroid per line, normalized to image size):
    class_id x_center y_center w h        # w/h are unused by FOMO

Usage:
    python3 evaluate-fomo.py
    python3 evaluate-fomo.py \
        --images <dir> --labels <dir> \
        --model "FOMO-FP32=models/fomo_fp32.onnx" \
        --model "FOMO-INT8=models/fomo_int8.onnx"
"""

import argparse
import csv
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# Distance sweep (in grid cells) used for the mAP-like aggregate.
# Smaller = stricter match. 10 thresholds, mirroring the COCO IoU sweep size.
DISTANCE_THRESHOLDS = np.linspace(0.5, 5.0, 10)

# Keep a loose confidence while sweeping P/R curves.
CONF_THRESHOLD_INFER = 0.05

# Distance (in grid cells) and confidence used for the confusion matrix.
CONFMAT_DIST = 2.0
CONFMAT_CONF = 0.5

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

DEFAULT_MODELS = {
    "FOMO-FP32": "models/fomo_fp32.onnx",
    # "FOMO-INT8": "models/fomo_int8.onnx",
}
DEFAULT_VAL_IMAGES = (
    "/Users/besnn/PycharmProjects/YOLOv8 Traffic Light Model/"
    "traffic_signal_detection/src/datasets/split_centroid_dataset/test/images"
)
DEFAULT_VAL_LABELS = (
    "/Users/besnn/PycharmProjects/YOLOv8 Traffic Light Model/"
    "traffic_signal_detection/src/datasets/split_centroid_dataset/test/labels"
)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_gt(label_path: str, grid_w: int, grid_h: int):
    """Return list of (class_id, (gx, gy)) in grid-cell coordinates."""
    out = []
    if not os.path.isfile(label_path):
        return out
    with open(label_path) as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 3:
                continue
            cls = int(parts[0])
            x_c = float(parts[1])
            y_c = float(parts[2])
            gx = x_c * grid_w
            gy = y_c * grid_h
            out.append((cls, (gx, gy)))
    return out


# --------------------------------------------------------------------------- #
# FOMO inference
# --------------------------------------------------------------------------- #
def preprocess(image, input_size):
    """Resize to (input_size, input_size), scale to [0, 1], CHW float32.

    No letterboxing — FOMO is trained with a plain Resize, see training_fomo_v2.
    """
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(blob, 0)


def softmax(logits, axis):
    m = logits.max(axis=axis, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=axis, keepdims=True)


def cluster_active_cells(score_map, cls_map, conf_thres):
    """Group adjacent active cells (8-connectivity) per class.

    Returns one detection per cluster, taking the argmax-score cell as the
    centroid location. Score is the max score in the cluster.
    """
    h, w = score_map.shape
    active = score_map >= conf_thres
    visited = np.zeros_like(active)
    detections = []
    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]
    for y in range(h):
        for x in range(w):
            if not active[y, x] or visited[y, x]:
                continue
            cls = int(cls_map[y, x])
            queue = deque([(y, x)])
            visited[y, x] = True
            best_score = score_map[y, x]
            best_xy = (x, y)
            while queue:
                cy, cx = queue.popleft()
                for dy, dx in neigh:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        if active[ny, nx] and int(cls_map[ny, nx]) == cls:
                            visited[ny, nx] = True
                            if score_map[ny, nx] > best_score:
                                best_score = score_map[ny, nx]
                                best_xy = (nx, ny)
                            queue.append((ny, nx))
            # Centroid in grid coords: cell-center (+0.5)
            detections.append({
                "cell": (best_xy[0] + 0.5, best_xy[1] + 0.5),
                "cls": cls,
                "score": float(best_score),
            })
    return detections


def postprocess(raw, num_classes, conf_thres):
    """Decode FOMO output: (1, C, H, W) logits where C = num_classes + 1."""
    arr = np.asarray(raw)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected FOMO output shape: {arr.shape}")

    # ONNX exports keep the (C, H, W) layout from PyTorch.
    c = arr.shape[0]
    if c != num_classes + 1:
        # Some exports may transpose channels-last — try to recover.
        if arr.shape[-1] == num_classes + 1:
            arr = arr.transpose(2, 0, 1)
        else:
            raise ValueError(
                f"FOMO output has {arr.shape[0]} channels, expected {num_classes + 1}"
            )

    probs = softmax(arr, axis=0)         # (C, H, W)
    fg = probs[1:]                       # (num_classes, H, W)
    cls_map = fg.argmax(axis=0)          # 0..num_classes-1
    score_map = fg.max(axis=0)
    return cluster_active_cells(score_map, cls_map, conf_thres)


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


def centroid_distance_matrix(dets, gts):
    """Pairwise Euclidean distance (grid cells) between dets and gts."""
    if not dets or not gts:
        return np.zeros((len(dets), len(gts)), dtype=np.float32)
    d_xy = np.array([d["cell"] for d in dets], dtype=np.float32)        # (Nd, 2)
    g_xy = np.array([g[1] for g in gts], dtype=np.float32)              # (Ng, 2)
    diff = d_xy[:, None, :] - g_xy[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def match_for_map(dets, gts, ndist):
    """Return (N_det, ndist) bool TP array. Detections are already score-sorted.

    A detection is a TP at distance threshold `t` if it lies within `t` grid
    cells of an unmatched GT of the same class.
    """
    tp = np.zeros((len(dets), ndist), dtype=bool)
    if not dets or not gts:
        return tp
    dist_mat = centroid_distance_matrix(dets, gts)
    same_class = np.array(
        [[d["cls"] == gc for gc, _ in gts] for d in dets], dtype=bool
    )
    # Disable cross-class pairings up-front.
    dist_mat = np.where(same_class, dist_mat, np.inf)

    for ti, thr in enumerate(DISTANCE_THRESHOLDS):
        used = np.zeros(len(gts), dtype=bool)
        for di in range(len(dets)):
            row = dist_mat[di]
            best_d, best_gi = thr, -1
            for gi in range(len(gts)):
                if used[gi]:
                    continue
                if row[gi] <= best_d:
                    best_d = row[gi]
                    best_gi = gi
            if best_gi >= 0:
                tp[di, ti] = True
                used[best_gi] = True
    return tp


def update_confusion_matrix(cm, dets, gts, num_classes,
                            conf_thres=CONFMAT_CONF, dist_thres=CONFMAT_DIST):
    """Class-agnostic matching at `dist_thres`; record class confusion."""
    hi = [d for d in dets if d["score"] >= conf_thres]
    hi.sort(key=lambda d: -d["score"])
    if not hi and not gts:
        return

    if hi and gts:
        d_xy = np.array([d["cell"] for d in hi], dtype=np.float32)
        g_xy = np.array([g[1] for g in gts], dtype=np.float32)
        diff = d_xy[:, None, :] - g_xy[None, :, :]
        dist_mat = np.sqrt((diff * diff).sum(-1))
    else:
        dist_mat = None

    gt_used = [False] * len(gts)
    det_matched = [False] * len(hi)
    for di, d in enumerate(hi):
        if dist_mat is None:
            break
        best_d, best_gi = dist_thres, -1
        for gi in range(len(gts)):
            if gt_used[gi]:
                continue
            if dist_mat[di, gi] <= best_d:
                best_d = dist_mat[di, gi]
                best_gi = gi
        if best_gi >= 0:
            gc = gts[best_gi][0]
            cm[gc, d["cls"]] += 1
            gt_used[best_gi] = True
            det_matched[di] = True
    for di, d in enumerate(hi):
        if not det_matched[di]:
            cm[num_classes, d["cls"]] += 1
    for gi, (gc, _) in enumerate(gts):
        if not gt_used[gi]:
            cm[gc, num_classes] += 1


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
def summarize_high_conf_dets(dets, classes):
    counts = {}
    for d in dets:
        cname = classes[d["cls"]] if d["cls"] < len(classes) else str(d["cls"])
        counts[cname] = counts.get(cname, 0) + 1
    return "; ".join(f"{name}:{count}" for name, count in sorted(counts.items()))


def infer_grid_size(sess, input_size, in_name):
    """Probe the model to discover its output grid resolution."""
    dummy = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    out = sess.run(None, {in_name: dummy})[0]
    arr = np.asarray(out)
    if arr.ndim == 4:
        _, _, h, w = arr.shape
    elif arr.ndim == 3:
        _, h, w = arr.shape
    else:
        raise ValueError(f"Unexpected FOMO output shape: {arr.shape}")
    return int(w), int(h)


def evaluate(model_path, images_dir, labels_dir, providers, classes,
             ignore_empty_labels=False, confmat_conf=CONFMAT_CONF,
             confmat_dist=CONFMAT_DIST):
    sess = ort.InferenceSession(model_path, providers=providers)
    inp = sess.get_inputs()[0]
    in_name = inp.name
    # FOMO ONNX exports are static-shape: input shape is (1, 3, H, W).
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 480

    grid_w, grid_h = infer_grid_size(sess, input_size, in_name)

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
    t_total = time.perf_counter()
    for i, img_path in enumerate(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        gts = load_gt(str(label_path), grid_w, grid_h)
        if ignore_empty_labels and not gts:
            if (i + 1) % 10 == 0 or i == len(img_paths) - 1:
                print(f"  [{i + 1:>4}/{len(img_paths)}] {img_path.name} (skipped empty label)")
            continue

        blob = preprocess(img, input_size)

        t0 = time.perf_counter()
        outputs = sess.run(None, {in_name: blob})
        inf_times.append(time.perf_counter() - t0)
        evaluated_images += 1

        dets = postprocess(outputs[0], nc, CONF_THRESHOLD_INFER)
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

        tp_arr = match_for_map(dets, gts, ndist)
        for di, d in enumerate(dets):
            stats_tp[d["cls"]].append(tp_arr[di])
            stats_conf[d["cls"]].append(d["score"])

        update_confusion_matrix(cm, dets, gts, nc, confmat_conf, confmat_dist)

        if (i + 1) % 10 == 0 or i == len(img_paths) - 1:
            print(f"  [{i + 1:>4}/{len(img_paths)}] {img_path.name}")

    wall = time.perf_counter() - t_total

    ap = np.zeros((nc, ndist), dtype=np.float64)
    p_best = np.zeros(nc)
    r_best = np.zeros(nc)
    f1_best = np.zeros(nc)
    thr_best = np.zeros(nc)

    # Primary distance threshold = the strictest in the sweep (index 0).
    primary_ti = 0

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
    print(f"\n--- {name} per-class metrics (distance={primary:.1f} cells for P/R/F1) ---")
    header = (
        f"{'class':<22}{'GT':>6}{'P':>10}{'R':>10}{'F1':>10}"
        f"{'AP@d=' + format(primary, '.1f'):>12}{'AP@d=sweep':>12}{'conf*':>10}"
    )
    print(header)
    print("-" * len(header))
    for c, name_c in enumerate(classes):
        ap_primary = ap[c, 0]
        ap_mean = ap[c].mean()
        print(
            f"{name_c:<22}{int(n_gt[c]):>6}"
            f"{result['precision'][c]:>10.4f}"
            f"{result['recall'][c]:>10.4f}"
            f"{result['f1'][c]:>10.4f}"
            f"{ap_primary:>12.4f}"
            f"{ap_mean:>12.4f}"
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


def print_confusion_matrix(name, cm, classes, confmat_conf, confmat_dist):
    labels = list(classes) + ["background"]
    width = max(max(len(l) for l in labels), 10)
    print(
        f"\n--- {name} confusion matrix "
        f"(rows = GT, cols = prediction; dist≤{confmat_dist}, conf≥{confmat_conf}) ---"
    )
    header = " " * (width + 2) + "".join(f"{l:>{width + 2}}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        line = f"{row_label:<{width + 2}}" + "".join(
            f"{int(cm[i, j]):>{width + 2}}" for j in range(len(labels))
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
            f"AP@d={primary:.1f}", "AP@d=sweep_mean", "best_conf",
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
            ])
        w.writerow([
            "mean",
            int(result["n_gt"].sum()),
            f"{result['precision'].mean():.6f}",
            f"{result['recall'].mean():.6f}",
            f"{result['f1'].mean():.6f}",
            f"{result['ap_primary']:.6f}",
            f"{result['ap_mean']:.6f}",
            "",
        ])
        w.writerow([])
        w.writerow(["distance_thresholds", ",".join(f"{t:.2f}" for t in result["distance_thresholds"])])
        w.writerow(["grid_size", f"{result['grid_size'][0]}x{result['grid_size'][1]}"])
        w.writerow(["inference_ms_mean", f"{result['inference_ms_mean']:.4f}"])
        w.writerow(["inference_ms_p95", f"{result['inference_ms_p95']:.4f}"])
        w.writerow(["wall_seconds", f"{result['wall_seconds']:.4f}"])
        w.writerow(["num_images", result["num_images"]])
        w.writerow(["input_size", result["input_size"]])


def save_metrics_png(path, name, result):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping {path}: {exc})")
        return

    classes = result["classes"]
    primary = result["primary_distance"]
    columns = [
        "class", "precision", "recall", "F1",
        f"AP@d={primary:.1f}", "AP@d=sweep", "best_conf",
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
            f"{result['conf_thr'][c]:.4f}",
        ])
    rows.append([
        "mean",
        f"{result['precision'].mean():.4f}",
        f"{result['recall'].mean():.4f}",
        f"{result['f1'].mean():.4f}",
        f"{result['ap_primary']:.4f}",
        f"{result['ap_mean']:.4f}",
        "",
    ])

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
    ap.add_argument("--model", action="append", default=None,
                    metavar="NAME=PATH",
                    help="Model entry (repeatable). Defaults to the bundled "
                         "FOMO FP32 ONNX model.")
    ap.add_argument("--output-dir", default="benchmark_results/fomo_eval",
                    help="Where to write CSV / PNG outputs.")
    ap.add_argument("--providers", nargs="+",
                    default=["CPUExecutionProvider"],
                    help="ONNX Runtime execution providers.")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                    help="Class names in order (must match training).")
    ap.add_argument("--ignore-empty-labels", action="store_true",
                    help="Skip images with no GT centroids.")
    ap.add_argument("--confmat-conf", type=float, default=CONFMAT_CONF,
                    help="Confidence threshold for the confusion matrix only. "
                         "AP/P/R/F1 are still computed from the full confidence sweep.")
    ap.add_argument("--confmat-dist", type=float, default=CONFMAT_DIST,
                    help="Distance (grid cells) threshold for confusion matrix matching.")
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
