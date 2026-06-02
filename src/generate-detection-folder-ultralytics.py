#!/usr/bin/env python3
"""Generate an inspection-folder JSON using the Ultralytics YOLO pipeline.

Like generate-detection-folder.py but uses `ultralytics.YOLO` for inference
and NMS instead of raw ONNX Runtime + custom postprocessing. The output
inspection.json is identical and works with the same error-inspection-app.

Usage:
    python3 generate-detection-folder-ultralytics.py \
        --model "YOLOv8n-INT8=models/yolov8n/yolov8n-obb-int8.onnx" \
        --images datasets/yolo_pl_test/images \
        --labels datasets/yolo_pl_test/labels
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "benchmark_results/detection_folders"
DEFAULT_IMAGES = SCRIPT_DIR / "datasets/yolo_pl_test/images"
DEFAULT_LABELS = SCRIPT_DIR / "datasets/yolo_pl_test/labels"
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

# Keep all boxes through NMS for evaluation, re-filter at CONFMAT_CONF for matching.
DEFAULT_CONF_INFER = 0.001
DEFAULT_NMS_IOU = 0.1
DEFAULT_CONFMAT_CONF = 0.5
DEFAULT_CONFMAT_IOU = 0.45


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
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# GT loading
# --------------------------------------------------------------------------- #
def load_gt(label_path: Path, img_w: int, img_h: int) -> tuple[list[tuple], list[str]]:
    """Load a YOLO-OBB label file → list of (cls_id, poly_pixels), issues."""
    gts, issues = [], []
    if not label_path.is_file():
        return gts, issues
    for lineno, raw in enumerate(label_path.read_text().splitlines(), start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) < 9:
            issues.append(f"line {lineno}: expected 9 values, got {len(parts)}")
            continue
        try:
            cls = int(float(parts[0]))
            xy = np.array(parts[1:9], dtype=np.float64).reshape(4, 2)
        except ValueError as exc:
            issues.append(f"line {lineno}: parse error: {exc}")
            continue
        if not np.isfinite(xy).all():
            issues.append(f"line {lineno}: non-finite coordinates")
            continue
        if (xy < 0).any() or (xy > 1).any():
            issues.append(f"line {lineno}: coordinates outside [0, 1]")
        xy[:, 0] = np.clip(xy[:, 0], 0.0, 1.0) * img_w
        xy[:, 1] = np.clip(xy[:, 1], 0.0, 1.0) * img_h
        gts.append((cls, xy.astype(np.float32)))
    return gts, issues


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def obb_result_to_dets(result) -> list[dict]:
    """Convert an Ultralytics OBB result to det-dicts used by match_errors."""
    obb = result.obb
    if obb is None or len(obb) == 0:
        return []
    polys = obb.xyxyxyxy.cpu().numpy()   # (N, 4, 2) absolute pixel coords
    confs = obb.conf.cpu().numpy()        # (N,)
    classes = obb.cls.cpu().numpy()       # (N,)
    return [
        {"cls": int(classes[i]), "score": float(confs[i]), "poly": polys[i]}
        for i in range(len(confs))
    ]


# --------------------------------------------------------------------------- #
# TP/FP/FN matching
# --------------------------------------------------------------------------- #
def match_errors(dets, gts, conf_thres, iou_thres):
    high_dets = [d for d in dets if d["score"] >= conf_thres]
    det_matches = [None] * len(high_dets)
    gt_matches = [None] * len(gts)
    best_iou_by_det = [0.0] * len(high_dets)

    candidates = []
    for di, det in enumerate(high_dets):
        for gi, (_cls, gt_poly) in enumerate(gts):
            iou = poly_iou(det["poly"], gt_poly)
            best_iou_by_det[di] = max(best_iou_by_det[di], iou)
            if iou >= iou_thres:
                candidates.append((iou, di, gi))

    candidates.sort(reverse=True, key=lambda x: x[0])
    used_dets, used_gts = set(), set()
    for iou, di, gi in candidates:
        if di in used_dets or gi in used_gts:
            continue
        det_matches[di] = (gi, iou)
        gt_matches[gi] = (di, iou)
        used_dets.add(di)
        used_gts.add(gi)

    det_status, gt_status = [], []
    has_error = False
    for di, det in enumerate(high_dets):
        m = det_matches[di]
        if m is None:
            det_status.append(("FP", None, best_iou_by_det[di]))
            has_error = True
        else:
            gi, iou = m
            if det["cls"] == gts[gi][0]:
                det_status.append(("TP", gi, iou))
            else:
                det_status.append(("CLASS_MISMATCH", gi, iou))
                has_error = True

    for gi, (gt_cls, _) in enumerate(gts):
        m = gt_matches[gi]
        if m is None:
            gt_status.append(("FN", None, 0.0))
            has_error = True
        else:
            di, iou = m
            if high_dets[di]["cls"] == gt_cls:
                gt_status.append(("TP", di, iou))
            else:
                gt_status.append(("CLASS_MISMATCH", di, iou))

    counts = {
        "fp": sum(1 for s, _, _ in det_status if s == "FP"),
        "fn": sum(1 for s, _, _ in gt_status if s == "FN"),
        "class_mismatch": sum(1 for s, _, _ in det_status if s == "CLASS_MISMATCH"),
        "tp": sum(1 for s, _, _ in det_status if s == "TP"),
    }
    return high_dets, det_status, gt_status, counts, has_error


# --------------------------------------------------------------------------- #
# JSON serialisation
# --------------------------------------------------------------------------- #
def class_name(classes: list[str], cls_id: int) -> str:
    return classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)


def poly_to_json(poly: np.ndarray) -> list[dict]:
    return [{"x": float(x), "y": float(y)} for x, y in poly.tolist()]


def detection_to_json(det: dict, status: tuple, classes: list[str], gts: list[tuple]) -> dict:
    state, match_idx, match_score = status
    ious, best = [], {"index": None, "iou": 0.0}
    for gi, (gt_cls, gt_poly) in enumerate(gts):
        if int(det["cls"]) != int(gt_cls):
            continue
        iou = float(poly_iou(det["poly"], gt_poly))
        ious.append({"gtIndex": gi, "classId": int(gt_cls),
                     "className": class_name(classes, int(gt_cls)), "iou": iou})
        if iou > best["iou"]:
            best = {"index": gi, "iou": iou}
    return {
        "classId": int(det["cls"]),
        "className": class_name(classes, int(det["cls"])),
        "score": float(det["score"]),
        "polygon": poly_to_json(det["poly"]),
        "status": state,
        "matchIndex": match_idx,
        "matchScore": float(match_score),
        "bestIoU": float(best["iou"]),
        "bestIoUIndex": best["index"],
        "ious": ious,
    }


def gt_to_json(gt: tuple, status: tuple, classes: list[str]) -> dict:
    cls_id, poly = gt
    state, match_idx, score = status
    return {
        "classId": int(cls_id),
        "className": class_name(classes, int(cls_id)),
        "polygon": poly_to_json(poly),
        "status": state,
        "matchIndex": match_idx,
        "matchScore": float(score),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="NAME=PATH",
                        help="Model to run, e.g. YOLOv8n-INT8=models/yolov8n-obb-int8.onnx")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--no-labels", action="store_true",
                        help="Skip GT loading; every detection is marked DET.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--conf-infer", type=float, default=DEFAULT_CONF_INFER,
                        help="Confidence threshold passed to Ultralytics (pre-NMS filter).")
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU,
                        help="IoU threshold for Ultralytics NMS.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFMAT_CONF,
                        help="Confidence threshold for TP/FP/FN assignment.")
    parser.add_argument("--iou", type=float, default=DEFAULT_CONFMAT_IOU,
                        help="IoU threshold for TP/FP/FN assignment.")
    parser.add_argument("--copy-raw", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def parse_model(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit("--model expects NAME=PATH")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def slugify(name: str) -> str:
    return "-".join(p for p in "".join(c if c.isalnum() else "-" for c in name.lower()).split("-") if p)


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        raise SystemExit("ultralytics is not installed in this environment.")

    name, model_path_raw = parse_model(args.model)
    model_path = Path(model_path_raw)
    if not model_path.is_absolute():
        model_path = SCRIPT_DIR / model_path
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}")
    if not args.images.is_dir():
        raise SystemExit(f"Images directory not found: {args.images}")
    if not args.no_labels and not args.labels.is_dir():
        raise SystemExit(f"Labels directory not found: {args.labels}")

    output_dir = args.output_root / slugify(name)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {name}: {model_path}")
    model = YOLO(str(model_path))

    img_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    records = []
    totals = {"fp": 0, "fn": 0, "mismatch": 0, "tp": 0, "detections": 0, "processed": 0}

    for idx, img_path in enumerate(img_paths, start=1):
        result = model(str(img_path), conf=args.conf_infer, iou=args.nms_iou, verbose=False)[0]

        h, w = result.orig_shape
        dets = obb_result_to_dets(result)
        dets.sort(key=lambda d: -d["score"])

        gts, label_issues = [], []
        if not args.no_labels:
            gts, label_issues = load_gt(args.labels / f"{img_path.stem}.txt", w, h)

        if args.no_labels:
            high_dets = [d for d in dets if d["score"] >= args.conf]
            det_status = [("DET", None, float(d["score"])) for d in high_dets]
            gt_status = []
            counts = {"fp": 0, "fn": 0, "class_mismatch": 0, "tp": 0}
            has_error = False
        else:
            high_dets, det_status, gt_status, counts, has_error = match_errors(
                dets, gts, args.conf, args.iou
            )
        counts["detections"] = len(high_dets)

        if args.copy_raw:
            shutil.copy2(img_path, raw_dir / img_path.name)

        records.append({
            "image": img_path.name,
            "width": w,
            "height": h,
            "rawImage": f"raw_images/{img_path.name}" if args.copy_raw else str(img_path),
            "annotatedImage": "",
            "hasError": has_error,
            "counts": counts,
            "labelIssues": label_issues,
            "groundTruth": [gt_to_json(gt, gt_status[gi], args.classes) for gi, gt in enumerate(gts)],
            "detections": [detection_to_json(det, det_status[di], args.classes, gts) for di, det in enumerate(high_dets)],
        })

        totals["processed"] += 1
        totals["fp"] += int(counts.get("fp", 0))
        totals["fn"] += int(counts.get("fn", 0))
        totals["mismatch"] += int(counts.get("class_mismatch", 0))
        totals["tp"] += int(counts.get("tp", 0))
        totals["detections"] += len(high_dets)

        if idx % 25 == 0 or idx == len(img_paths):
            print(f"  [{idx:>4}/{len(img_paths)}] detections={totals['detections']}")

    payload = {
        "model": {"name": name, "path": model_path_raw, "kind": "ultralytics"},
        "thresholds": {"confidence": args.conf, "iou": args.iou,
                       "confInfer": args.conf_infer, "nmsIou": args.nms_iou},
        "classes": args.classes,
        "totals": totals,
        "records": records,
    }
    out = output_dir / "inspection.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[OK] Wrote {out}")
    print(f"     FP={totals['fp']} FN={totals['fn']} mismatch={totals['mismatch']} TP={totals['tp']}")
    print(f"Open with error-inspection-app using --results-root {args.output_root}")


if __name__ == "__main__":
    main()
