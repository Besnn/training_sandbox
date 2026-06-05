#!/usr/bin/env python3
"""Generate a self-contained inspection folder for FOMO detections.

FOMO predicts centroids rather than boxes, so the JSON layers use a centroid
(plus the grid cell) instead of a 4-vertex polygon, and matching is by
distance in output-grid cells instead of polygon IoU.

This runs one FOMO ONNX model on a validation image folder and writes:
  - raw_images/<image>
  - annotated_images/<image>__fpX_fnY_mismatchZ.jpg
  - error_inspection/*.jpg                 compatibility copy for the web app
  - error_inspection/error_report.csv      compatibility CSV
  - inspection.json                        full GT/detection/status layers
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "benchmark_results/inspection_folders"

cv2 = None
np = None
ort = None
FOMO = None
FOMO_INSPECT = None


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_modules() -> None:
    global cv2, np, ort, FOMO, FOMO_INSPECT

    try:
        import cv2 as cv2_module
        import numpy as np_module
        import onnxruntime as ort_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing inference dependency: {exc.name}. Run this script in the "
            "same environment used for the benchmark evaluators."
        ) from exc

    cv2 = cv2_module
    np = np_module
    ort = ort_module
    FOMO = load_module("evaluate-fomo.py", "evaluate_fomo")
    FOMO_INSPECT = load_module("inspect-fomo-errors.py", "inspect_fomo_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="NAME=PATH",
                        help="Model to inspect, for example FOMO-FP32=models/fomo-480-onnx/fomo-480.onnx.")
    parser.add_argument("--images", default=None,
                        help="Directory of validation images (defaults to evaluate-fomo.DEFAULT_VAL_IMAGES).")
    parser.add_argument("--labels", default=None,
                        help="Directory of YOLO-centroid label files (defaults to evaluate-fomo.DEFAULT_VAL_LABELS).")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Root folder for generated inspection folders.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime execution providers.")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Class names in label order (defaults to evaluate-fomo.DEFAULT_CLASSES).")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence used for TP/FP/FN assignment.")
    parser.add_argument("--dist", type=float, default=None,
                        help="Distance threshold (grid cells) used for TP/FP/FN assignment.")
    parser.add_argument("--all-images", action="store_true",
                        help="Save annotated outputs for every processed image, not only images with errors.")
    parser.add_argument("--copy-raw", action="store_true",
                        help="Copy raw validation images into raw_images/.")
    parser.add_argument("--ignore-empty-labels", action="store_true",
                        help="Skip images with no GT centroids.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional quick-test image limit.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the output folder before writing.")
    return parser.parse_args()


def parse_model(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit("--model expects NAME=PATH")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        out.append(ch if ch.isalnum() else "-")
    return "-".join(part for part in "".join(out).split("-") if part)


def class_name(classes: list[str], cls_id: int) -> str:
    return classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)


def cell_to_pixel(cell, img_w: int, img_h: int, grid_w: int, grid_h: int) -> tuple[float, float]:
    """Map a centroid to raw-image pixels, accounting for FOMO's input crop.

    NOTE: `cell` is already NORMALIZED to [0,1] (both evaluate-fomo's postprocess
    and load_gt return normalized centroids, not raw grid indices). So we scale by
    the (cropped) image size directly — dividing by grid_w/grid_h here would
    normalize a second time and collapse every point into the top-left corner.
    grid_w/grid_h are kept only for signature compatibility with the callers.
    """
    crop_left = getattr(FOMO, "CROP_LEFT", 0)
    crop_bottom = getattr(FOMO, "CROP_BOTTOM", 0)
    cropped_w = max(1, img_w - crop_left)
    cropped_h = max(1, img_h - crop_bottom)
    return (
        crop_left + cell[0] * cropped_w,
        cell[1] * cropped_h,
    )


def centroid_to_json(cell, img_w: int, img_h: int, grid_w: int, grid_h: int) -> dict[str, float]:
    px, py = cell_to_pixel(cell, img_w, img_h, grid_w, grid_h)
    return {
        "x": float(px),
        "y": float(py),
        "cellX": float(cell[0]),
        "cellY": float(cell[1]),
    }


def detection_distances_to_json(det: dict, gts: list[tuple], classes: list[str]) -> tuple[list[dict], dict]:
    distances = []
    best = {"index": None, "distance": float("inf")}

    for gi, (gt_cls, gt_cell) in enumerate(gts):
        if int(det["cls"]) != int(gt_cls):
            continue
        dist = float(FOMO_INSPECT.centroid_distance(det, (gt_cls, gt_cell)))
        distances.append({
            "gtIndex": gi,
            "classId": int(gt_cls),
            "className": class_name(classes, int(gt_cls)),
            "distance": dist,
        })
        if dist < best["distance"]:
            best["index"] = gi
            best["distance"] = dist

    if best["index"] is None:
        best["distance"] = 0.0
    return distances, best


def detection_to_json(
    det: dict,
    status: tuple,
    classes: list[str],
    gts: list[tuple],
    img_w: int,
    img_h: int,
    grid_w: int,
    grid_h: int,
) -> dict:
    state, match_idx, score = status
    distances, best = detection_distances_to_json(det, gts, classes)
    return {
        "classId": int(det["cls"]),
        "className": class_name(classes, int(det["cls"])),
        "score": float(det["score"]),
        "centroid": centroid_to_json(det["cell"], img_w, img_h, grid_w, grid_h),
        "status": state,
        "matchIndex": match_idx,
        "matchDistance": float(score),
        "bestDistance": float(best["distance"]),
        "bestDistanceIndex": best["index"],
        "distances": distances,
    }


def gt_to_json(
    gt: tuple,
    status: tuple,
    classes: list[str],
    img_w: int,
    img_h: int,
    grid_w: int,
    grid_h: int,
) -> dict:
    cls_id, cell = gt
    state, match_idx, score = status
    return {
        "classId": int(cls_id),
        "className": class_name(classes, int(cls_id)),
        "centroid": centroid_to_json(cell, img_w, img_h, grid_w, grid_h),
        "status": state,
        "matchIndex": match_idx,
        "matchDistance": float(score),
    }


def write_report(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "annotated_image", "label_file", "gt_count",
            "false_positives", "false_negatives", "class_mismatches",
            "true_positives", "fp_scores",
        ])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    load_runtime_modules()

    images_dir = args.images if args.images is not None else FOMO.DEFAULT_VAL_IMAGES
    labels_dir = args.labels if args.labels is not None else FOMO.DEFAULT_VAL_LABELS
    classes = args.classes if args.classes is not None else list(FOMO.DEFAULT_CLASSES)

    name, model_path_raw = parse_model(args.model)
    model_path = FOMO.resolve_model_path(model_path_raw)
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not os.path.isdir(labels_dir):
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    output_dir = args.output_root / slugify(name)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw_images"
    annotated_dir = output_dir / "annotated_images"
    compat_dir = output_dir / "error_inspection"
    for path in (raw_dir, annotated_dir, compat_dir):
        path.mkdir(parents=True, exist_ok=True)

    conf_thres = args.conf if args.conf is not None else FOMO.CONFMAT_CONF
    dist_thres = args.dist if args.dist is not None else FOMO.CONFMAT_DIST

    print(f"Inspecting {name}: {model_path}")
    session = ort.InferenceSession(model_path, providers=args.providers)
    inp = session.get_inputs()[0]
    input_name = inp.name
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 480
    grid_w, grid_h = FOMO.infer_grid_size(session, input_size, input_name)

    img_paths = sorted(
        p for p in Path(images_dir).iterdir() if p.suffix.lower() in FOMO.IMG_EXTS
    )
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    records = []
    report_rows = []
    totals = {"fp": 0, "fn": 0, "mismatch": 0, "tp": 0, "saved": 0, "processed": 0}

    for idx, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = Path(labels_dir) / f"{img_path.stem}.txt"
        gts = FOMO.load_gt(str(label_path), grid_w, grid_h, raw_w=w, raw_h=h)
        if args.ignore_empty_labels and not gts:
            continue

        blob = FOMO.preprocess(img, input_size)
        outputs = session.run(None, {input_name: blob})
        dets = FOMO.postprocess(outputs[0], len(classes), FOMO.CONF_THRESHOLD_INFER)
        dets.sort(key=lambda d: -d["score"])

        high_dets, det_status, gt_status, counts, has_error = FOMO_INSPECT.match_errors(
            dets, gts, conf_thres, dist_thres
        )
        should_save = args.all_images or has_error
        annotated_name = ""
        if should_save:
            annotated_name = (
                f"{img_path.stem}__fp{counts['fp']}_fn{counts['fn']}"
                f"_mismatch{counts['class_mismatch']}.jpg"
            )
            rendered = FOMO_INSPECT.render_error_image(
                img, classes, gts, high_dets, det_status, gt_status, counts,
                (grid_w, grid_h), {"conf": conf_thres, "dist": dist_thres},
            )
            cv2.imwrite(str(annotated_dir / annotated_name), rendered)
            shutil.copy2(annotated_dir / annotated_name, compat_dir / annotated_name)
            if args.copy_raw:
                shutil.copy2(img_path, raw_dir / img_path.name)
            totals["saved"] += 1

            fp_scores = [
                f"{high_dets[di]['score']:.4f}"
                for di, (status, _match_idx, _score) in enumerate(det_status)
                if status == "FP"
            ]
            report_rows.append([
                img_path.name,
                annotated_name,
                str(label_path),
                len(gts),
                counts["fp"],
                counts["fn"],
                counts["class_mismatch"],
                counts["tp"],
                ";".join(fp_scores),
            ])

        records.append({
            "image": img_path.name,
            "width": w,
            "height": h,
            "rawImage": f"raw_images/{img_path.name}" if args.copy_raw and should_save else str(img_path),
            "annotatedImage": f"annotated_images/{annotated_name}" if annotated_name else "",
            "hasError": has_error,
            "counts": counts,
            "groundTruth": [
                gt_to_json(gt, gt_status[gi], classes, w, h, grid_w, grid_h)
                for gi, gt in enumerate(gts)
            ],
            "detections": [
                detection_to_json(det, det_status[di], classes, gts, w, h, grid_w, grid_h)
                for di, det in enumerate(high_dets)
            ],
        })

        totals["processed"] += 1
        totals["fp"] += counts["fp"]
        totals["fn"] += counts["fn"]
        totals["mismatch"] += counts["class_mismatch"]
        totals["tp"] += counts["tp"]

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"  [{idx:>4}/{len(img_paths)}] saved {totals['saved']} inspection images")

    write_report(compat_dir / "error_report.csv", report_rows)
    payload = {
        "model": {
            "name": name,
            "path": model_path,
            "architecture": "fomo",
            "inputSize": input_size,
            "gridWidth": grid_w,
            "gridHeight": grid_h,
            "cropLeft": int(getattr(FOMO, "CROP_LEFT", 0)),
            "cropBottom": int(getattr(FOMO, "CROP_BOTTOM", 0)),
        },
        "thresholds": {
            "confidence": conf_thres,
            "distance": dist_thres,
        },
        "classes": classes,
        "totals": totals,
        "records": records,
    }
    (output_dir / "inspection.json").write_text(json.dumps(payload, indent=2))

    print(
        f"[OK] Wrote {output_dir} "
        f"(saved={totals['saved']}, FP={totals['fp']}, FN={totals['fn']}, "
        f"mismatch={totals['mismatch']})"
    )


if __name__ == "__main__":
    main()
