#!/usr/bin/env python3
"""Generate a self-contained inspection folder for YOLO-OBB detections.

This runs one YOLO-OBB ONNX model on a validation image folder and writes:
  - raw_images/<image>
  - annotated_images/<image>__fpX_fnY_mismatchZ.jpg
  - error_inspection/*.jpg                 compatibility copy for the web app
  - error_inspection/error_report.csv      compatibility CSV
  - inspection.json                        full GT/detection/status layers

The JSON file is meant for interactive inspection UIs where GT, TP, FP, FN,
and class-mismatch overlays can be toggled independently.
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
DEFAULT_IMAGES = (
    "/home/arduino/ArduinoUnoQSandbox/"
    "examples/benchmarks/datasets/yolo_pl_test/images"
)
DEFAULT_LABELS = (
    "/home/arduino/ArduinoUnoQSandbox/"
    "examples/benchmarks/datasets/yolo_pl_test/labels"
)
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]

cv2 = None
np = None
ort = None
YOLO_FP32 = None
YOLO_INT8 = None
YOLO_INSPECT = None


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime_modules() -> None:
    global cv2, np, ort, YOLO_FP32, YOLO_INT8, YOLO_INSPECT

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
    YOLO_FP32 = load_module("evaluate-yolo-obb.py", "evaluate_yolo_obb")
    YOLO_INT8 = load_module("evaluate-yolo-obb-int8.py", "evaluate_yolo_obb_int8")
    YOLO_INSPECT = load_module("inspect-yolo-obb-errors.py", "inspect_yolo_obb_errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="NAME=PATH",
                        help="Model to inspect, for example YOLOv8n=models/yolov8n/yolov8n-obb-fp32.onnx.")
    parser.add_argument("--kind", choices=["fp32", "int8"], default="fp32",
                        help="Which evaluator/postprocessor to use.")
    parser.add_argument("--images", default=DEFAULT_IMAGES,
                        help="Directory of validation images.")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="Directory of YOLO-OBB labels.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Root folder for generated inspection folders.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime execution providers.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Class names in label order.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Detection confidence used for TP/FP/FN assignment.")
    parser.add_argument("--iou", type=float, default=None,
                        help="IoU threshold used for TP/FP/FN assignment.")
    parser.add_argument("--all-images", action="store_true",
                        help="Save annotated outputs for every processed image, not only images with errors.")
    parser.add_argument("--copy-raw", action="store_true",
                        help="Copy raw validation images into raw_images/.")
    parser.add_argument("--ignore-empty-labels", action="store_true",
                        help="Skip images with no GT labels.")
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


def resolve_model_path(module, path: str) -> str:
    model_path = Path(path)
    if model_path.is_absolute() or model_path.is_file():
        return str(model_path)
    script_relative = SCRIPT_DIR / model_path
    if script_relative.is_file():
        return str(script_relative)
    return str(model_path)


def make_session(module, model_path: str, providers: list[str]):
    session_model_path = model_path
    patched_output = False
    if hasattr(module, "use_pre_quantized_output"):
        session_model_path, patched_output = module.use_pre_quantized_output(model_path)
    if patched_output:
        print("[INFO] Using pre-quantized output tensor for INT8 inspection.")
    return ort.InferenceSession(session_model_path, providers=providers)


def poly_to_json(poly: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for x, y in np.asarray(poly).tolist()]


def class_name(classes: list[str], cls_id: int) -> str:
    return classes[cls_id] if 0 <= cls_id < len(classes) else str(cls_id)


def detection_ious_to_json(module, det: dict, gts: list[tuple], classes: list[str]) -> tuple[list[dict], dict]:
    ious = []
    best = {
        "index": None,
        "iou": 0.0,
    }

    for gi, (gt_cls, gt_poly) in enumerate(gts):
        if int(det["cls"]) != int(gt_cls):
            continue
        iou = float(module.poly_iou(det["poly"], gt_poly))
        ious.append({
            "gtIndex": gi,
            "classId": int(gt_cls),
            "className": class_name(classes, int(gt_cls)),
            "iou": iou,
        })
        if iou > best["iou"]:
            best["index"] = gi
            best["iou"] = iou

    return ious, best


def detection_to_json(module, det: dict, status: tuple, classes: list[str], gts: list[tuple]) -> dict:
    state, match_idx, score = status
    ious, best = detection_ious_to_json(module, det, gts, classes)
    return {
        "classId": int(det["cls"]),
        "className": class_name(classes, int(det["cls"])),
        "score": float(det["score"]),
        "polygon": poly_to_json(det["poly"]),
        "status": state,
        "matchIndex": match_idx,
        "matchScore": float(score),
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


def write_report(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "annotated_image", "label_file", "gt_count",
            "false_positives", "false_negatives", "class_mismatches",
            "true_positives", "fp_best_ious", "label_issues",
        ])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    load_runtime_modules()
    module = YOLO_INT8 if args.kind == "int8" else YOLO_FP32
    name, model_path_raw = parse_model(args.model)
    model_path = resolve_model_path(module, model_path_raw)
    if not os.path.isfile(model_path):
        raise SystemExit(f"Model not found: {model_path}")
    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    output_dir = args.output_root / slugify(name)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw_images"
    annotated_dir = output_dir / "annotated_images"
    compat_dir = output_dir / "error_inspection"
    for path in (raw_dir, annotated_dir, compat_dir):
        path.mkdir(parents=True, exist_ok=True)

    conf_thres = args.conf if args.conf is not None else module.CONFMAT_CONF
    iou_thres = args.iou if args.iou is not None else module.CONFMAT_IOU

    print(f"Inspecting {name}: {model_path}")
    session = make_session(module, model_path, args.providers)
    inp = session.get_inputs()[0]
    input_size = int(inp.shape[2])
    input_name = inp.name

    img_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in module.IMG_EXTS
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
        label_path = Path(args.labels) / f"{img_path.stem}.txt"
        gts, label_issues = module.load_gt(str(label_path), w, h, return_issues=True)
        if args.ignore_empty_labels and not gts:
            continue

        blob, meta = module.preprocess(img, input_size)
        outputs = session.run(None, {input_name: blob})
        dets = module.postprocess(
            outputs[0], input_size, w, h, len(args.classes),
            module.CONF_THRESHOLD_INFER, meta,
        )
        dets.sort(key=lambda d: -d["score"])

        high_dets, det_status, gt_status, counts, has_error = YOLO_INSPECT.match_errors(
            module, dets, gts, len(args.classes), conf_thres, iou_thres
        )
        should_save = args.all_images or has_error
        annotated_name = ""
        if should_save:
            annotated_name = (
                f"{img_path.stem}__fp{counts['fp']}_fn{counts['fn']}"
                f"_mismatch{counts['class_mismatch']}.jpg"
            )
            rendered = YOLO_INSPECT.render_error_image(
                img, args.classes, gts, high_dets, det_status, gt_status, counts,
                {"conf": conf_thres, "iou": iou_thres},
            )
            cv2.imwrite(str(annotated_dir / annotated_name), rendered)
            shutil.copy2(annotated_dir / annotated_name, compat_dir / annotated_name)
            if args.copy_raw:
                shutil.copy2(img_path, raw_dir / img_path.name)
            totals["saved"] += 1

            fp_best_ious = [
                f"{score:.4f}"
                for status, _match_idx, score in det_status
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
                ";".join(fp_best_ious),
                " | ".join(label_issues),
            ])

        records.append({
            "image": img_path.name,
            "width": w,
            "height": h,
            "rawImage": f"raw_images/{img_path.name}" if args.copy_raw and should_save else str(img_path),
            "annotatedImage": f"annotated_images/{annotated_name}" if annotated_name else "",
            "hasError": has_error,
            "counts": counts,
            "labelIssues": label_issues,
            "groundTruth": [
                gt_to_json(gt, gt_status[gi], args.classes)
                for gi, gt in enumerate(gts)
            ],
            "detections": [
                detection_to_json(module, det, det_status[di], args.classes, gts)
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
            "architecture": "yolo-obb",
            "kind": args.kind,
            "inputSize": input_size,
        },
        "thresholds": {
            "confidence": conf_thres,
            "iou": iou_thres,
        },
        "classes": args.classes,
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
