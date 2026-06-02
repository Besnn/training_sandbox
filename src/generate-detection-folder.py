#!/usr/bin/env python3
"""Generate an inspection-folder JSON containing detections for every image.

The output is meant for error-inspection-app:

    benchmark_results/inspection_folders/<model-name>/inspection.json

Unlike generate-inspection-folder.py, this script does not render annotated JPGs.
It writes detections/GT/status layers only, so the web app can draw overlays on
the raw image interactively.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "benchmark_results/detection_folders"
DEFAULT_IMAGES = SCRIPT_DIR / "datasets/yolo_pl_test/images"
DEFAULT_LABELS = SCRIPT_DIR / "datasets/yolo_pl_test/labels"
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, metavar="NAME=PATH",
                        help="Model to run, for example YOLOv8n=models/yolov8n/yolov8n-obb-fp32.onnx.")
    parser.add_argument("--kind", choices=["fp32", "int8"], default="fp32",
                        help="Which evaluator/postprocessor to use.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                        help="Directory of input images.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS,
                        help="Optional YOLO-OBB labels directory for TP/FP/FN matching.")
    parser.add_argument("--no-labels", action="store_true",
                        help="Do not load GT labels; every detection is marked DET.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Root output folder.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime execution providers.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Class names in label order.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Confidence threshold for detections included in inspection.json.")
    parser.add_argument("--iou", type=float, default=None,
                        help="IoU threshold used for TP/FP/FN assignment when labels are available.")
    parser.add_argument("--nms", action="store_true",
                        help="Apply evaluator NMS. By default this script keeps all decoded detections.")
    parser.add_argument("--copy-raw", action="store_true",
                        help="Copy raw images into the inspection folder.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional quick-test limit.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete the model output folder before writing.")
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
    if hasattr(module, "resolve_model_path"):
        return module.resolve_model_path(path)
    return str(model_path)


def blank_gt_status(gts: list[tuple]) -> list[tuple[str, int | None, float]]:
    return [("UNMATCHED", None, 0.0) for _ in gts]


def unmatched_det_status(dets: list[dict]) -> list[tuple[str, int | None, float]]:
    return [("DET", None, float(det.get("score", 0.0))) for det in dets]


def decode_without_nms(module, raw, input_size, img_w, img_h, num_classes, conf_thres, meta) -> list[dict]:
    original_rotated_nms = module.rotated_nms
    module.rotated_nms = lambda dets, _iou_thres: list(range(len(dets)))
    try:
        return module.postprocess(raw, input_size, img_w, img_h, num_classes, conf_thres, meta)
    finally:
        module.rotated_nms = original_rotated_nms


def main() -> None:
    args = parse_args()
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing inference dependency: {exc.name}. Run this in the same "
            "environment used for benchmark evaluation."
        ) from exc

    yolo_fp32 = load_module("evaluate-yolo-obb.py", "evaluate_yolo_obb")
    yolo_int8 = load_module("evaluate-yolo-obb-int8.py", "evaluate_yolo_obb_int8")
    inspect_gen = load_module("generate-inspection-folder.py", "generate_inspection_folder")
    inspect_match = load_module("inspect-yolo-obb-errors.py", "inspect_yolo_obb_errors")
    inspect_gen.np = np

    module = yolo_int8 if args.kind == "int8" else yolo_fp32
    name, model_path_raw = parse_model(args.model)
    model_path = resolve_model_path(module, model_path_raw)
    if not os.path.isfile(model_path):
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

    conf_thres = args.conf if args.conf is not None else module.CONFMAT_CONF
    iou_thres = args.iou if args.iou is not None else module.CONFMAT_IOU

    session_model_path = model_path
    patched_output = False
    if hasattr(module, "use_pre_quantized_output"):
        session_model_path, patched_output = module.use_pre_quantized_output(model_path)
    if patched_output:
        print("[INFO] Using pre-quantized output tensor for INT8 detections.")

    session = ort.InferenceSession(session_model_path, providers=args.providers)
    inp = session.get_inputs()[0]
    input_name = inp.name
    input_size = int(inp.shape[2])

    img_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in module.IMG_EXTS
    )
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    records = []
    totals = {"fp": 0, "fn": 0, "mismatch": 0, "tp": 0, "detections": 0, "processed": 0}

    for idx, img_path in enumerate(img_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        gts = []
        label_issues = []
        if not args.no_labels:
            label_path = args.labels / f"{img_path.stem}.txt"
            gts, label_issues = module.load_gt(str(label_path), w, h, return_issues=True)

        blob, meta = module.preprocess(img, input_size)
        outputs = session.run(None, {input_name: blob})
        if args.nms:
            dets = module.postprocess(
                outputs[0], input_size, w, h, len(args.classes),
                module.CONF_THRESHOLD_INFER, meta,
            )
        else:
            dets = decode_without_nms(
                module, outputs[0], input_size, w, h, len(args.classes),
                module.CONF_THRESHOLD_INFER, meta,
            )
        dets.sort(key=lambda det: -det["score"])
        high_dets = [det for det in dets if det["score"] >= conf_thres]

        if args.no_labels:
            det_status = unmatched_det_status(high_dets)
            gt_status = blank_gt_status(gts)
            counts = {
                "fp": 0,
                "fn": 0,
                "class_mismatch": 0,
                "tp": 0,
                "detections": len(high_dets),
            }
        else:
            high_dets, det_status, gt_status, counts, _has_error = inspect_match.match_errors(
                module, dets, gts, len(args.classes), conf_thres, iou_thres
            )
            counts["detections"] = len(high_dets)

        if args.copy_raw:
            shutil.copy2(img_path, raw_dir / img_path.name)
            raw_image = f"raw_images/{img_path.name}"
        else:
            raw_image = str(img_path)

        records.append({
            "image": img_path.name,
            "width": w,
            "height": h,
            "rawImage": raw_image,
            "annotatedImage": "",
            "hasError": bool(counts["fp"] or counts["fn"] or counts["class_mismatch"]),
            "counts": counts,
            "labelIssues": label_issues,
            "groundTruth": [
                inspect_gen.gt_to_json(gt, gt_status[gi], args.classes)
                for gi, gt in enumerate(gts)
            ],
            "detections": [
                inspect_gen.detection_to_json(module, det, det_status[di], args.classes, gts)
                for di, det in enumerate(high_dets)
            ],
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
        "model": {
            "name": name,
            "path": model_path_raw,
            "kind": args.kind,
            "inputSize": input_size,
        },
        "thresholds": {
            "confidence": conf_thres,
            "iou": iou_thres,
            "nms": bool(args.nms),
        },
        "classes": args.classes,
        "totals": totals,
        "records": records,
    }
    (output_dir / "inspection.json").write_text(json.dumps(payload, indent=2))
    print(f"[OK] Wrote {output_dir / 'inspection.json'}")
    print(f"Open with error-inspection-app using --results-root {args.output_root}")


if __name__ == "__main__":
    main()
