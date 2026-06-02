#!/usr/bin/env python3
"""Render YOLO-OBB false positives/false negatives for visual inspection.

Runs the listed YOLO FP32 and YOLO INT8 evaluators, matches predictions to
ground truth using the same confusion-matrix thresholds, and writes annotated
images for any image with a false positive, false negative, or class mismatch.

Output layout:
    benchmark_results/<model-name>/error_inspection/*.jpg
"""

import argparse
import csv
import importlib.util
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = str(SCRIPT_DIR / "benchmark_results")


def load_eval_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


YOLO_FP32 = load_eval_module("evaluate-yolo-obb.py", "evaluate_yolo_obb")
YOLO_INT8 = load_eval_module("evaluate-yolo-obb-int8.py", "evaluate_yolo_obb_int8")


def slugify(name):
    out = []
    for ch in name.lower():
        out.append(ch if ch.isalnum() else "-")
    return "-".join(part for part in "".join(out).split("-") if part)


def parse_model_args(items):
    out = {}
    for raw in items or []:
        if "=" not in raw:
            raise SystemExit(f"--model expects NAME=PATH, got {raw!r}")
        name, path = raw.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def resolve_model_path(module, path):
    if hasattr(module, "resolve_model_path"):
        return module.resolve_model_path(path)
    model_path = Path(path)
    if model_path.is_absolute() or model_path.is_file():
        return str(model_path)
    script_relative = SCRIPT_DIR / model_path
    if script_relative.is_file():
        return str(script_relative)
    return str(model_path)


def make_session(module, model_path, providers):
    session_model_path = model_path
    patched_output = False
    if hasattr(module, "use_pre_quantized_output"):
        session_model_path, patched_output = module.use_pre_quantized_output(model_path)
    if patched_output:
        print("  [INFO] Using pre-quantized output tensor for INT8 inspection.")
    session = ort.InferenceSession(session_model_path, providers=providers)
    return session


def match_errors(module, dets, gts, num_classes, conf_thres, iou_thres):
    high_dets = [d for d in dets if d["score"] >= conf_thres]
    det_matches = [None] * len(high_dets)
    gt_matches = [None] * len(gts)
    best_iou_by_det = [0.0] * len(high_dets)

    candidate_matches = []
    for di, det in enumerate(high_dets):
        for gi, (_gt_cls, gt_poly) in enumerate(gts):
            iou = module.poly_iou(det["poly"], gt_poly)
            best_iou_by_det[di] = max(best_iou_by_det[di], iou)
            if iou >= iou_thres:
                candidate_matches.append((iou, di, gi))

    # Pair detections and GTs by best overlap first. This avoids a merely
    # higher-confidence detection claiming a GT that another detection fits better.
    candidate_matches.sort(reverse=True, key=lambda item: item[0])
    used_dets = set()
    used_gts = set()
    for iou, di, gi in candidate_matches:
        if di in used_dets or gi in used_gts:
            continue
        det_matches[di] = (gi, iou)
        gt_matches[gi] = (di, iou)
        used_dets.add(di)
        used_gts.add(gi)

    det_status = []
    gt_status = []
    has_error = False

    for di, det in enumerate(high_dets):
        match = det_matches[di]
        if match is None:
            det_status.append(("FP", None, best_iou_by_det[di]))
            has_error = True
            continue
        gi, iou = match
        gt_cls = gts[gi][0]
        if det["cls"] == gt_cls:
            det_status.append(("TP", gi, iou))
        else:
            det_status.append(("CLASS_MISMATCH", gi, iou))
            has_error = True

    for gi, (gt_cls, _gt_poly) in enumerate(gts):
        match = gt_matches[gi]
        if match is None:
            gt_status.append(("FN", None, 0.0))
            has_error = True
            continue
        di, iou = match
        det_cls = high_dets[di]["cls"]
        if det_cls == gt_cls:
            gt_status.append(("TP", di, iou))
        else:
            gt_status.append(("CLASS_MISMATCH", di, iou))

    counts = {
        "fp": sum(1 for status, _, _ in det_status if status == "FP"),
        "fn": sum(1 for status, _, _ in gt_status if status == "FN"),
        "class_mismatch": sum(1 for status, _, _ in det_status if status == "CLASS_MISMATCH"),
        "tp": sum(1 for status, _, _ in det_status if status == "TP"),
    }
    return high_dets, det_status, gt_status, counts, has_error


def class_name(classes, cls_id):
    if 0 <= int(cls_id) < len(classes):
        return classes[int(cls_id)]
    return str(cls_id)


def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(0, min(int(x), img.shape[1] - tw - 4))
    y = max(th + 6, min(int(y), img.shape[0] - 4))
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y + baseline), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def label_anchor_outside_poly(img, poly, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_w = tw + 4
    label_h = th + baseline + 4
    pad = 5

    xs = poly[:, 0]
    ys = poly[:, 1]
    min_x, max_x = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    min_y, max_y = int(np.floor(ys.min())), int(np.ceil(ys.max()))

    candidates = [
        (min_x, min_y - pad, "above"),
        (min_x, max_y + label_h + pad, "below"),
        (max_x + pad, min_y + label_h, "right"),
        (min_x - label_w - pad, min_y + label_h, "left"),
    ]

    for x, y, side in candidates:
        rect_x1 = x
        rect_x2 = x + label_w
        rect_y1 = y - label_h
        rect_y2 = y
        if rect_x1 < 0 or rect_y1 < 0 or rect_x2 >= img.shape[1] or rect_y2 >= img.shape[0]:
            continue
        return x, y

    # Edge fallback: keep labels inside the image, preferring space above the box.
    x = max(0, min(min_x, img.shape[1] - label_w))
    if min_y - pad - label_h >= 0:
        y = min_y - pad
    elif max_y + label_h + pad < img.shape[0]:
        y = max_y + label_h + pad
    else:
        y = max(label_h, min(min_y, img.shape[0] - 4))
    return x, y


def draw_poly(img, poly, color, label, thickness=2):
    pts = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    x, y = label_anchor_outside_poly(img, poly, label)
    draw_label(img, label, x, y, color)


def render_error_image(img, classes, gts, dets, det_status, gt_status, counts, thresholds):
    canvas = img.copy()

    colors = {
        "TP_GT": (85, 150, 85),
        "TP_DET": (150, 170, 80),
        "FP": (30, 30, 230),
        "FN": (0, 145, 255),
        "CLASS_MISMATCH": (0, 210, 255),
    }

    # Draw predictions first, then GT on top. Otherwise a prediction outline can
    # hide the ground-truth polygon and make it look like the label was missing.
    for di, det in enumerate(dets):
        status, match_idx, iou = det_status[di]
        cname = class_name(classes, det["cls"])
        if status == "FP":
            label = f"FP pred {cname} {det['score']:.2f} IoU {iou:.2f}"
            draw_poly(canvas, det["poly"], colors["FP"], label, thickness=3)
        elif status == "CLASS_MISMATCH":
            label = f"Wrong pred {cname} {det['score']:.2f} IoU {iou:.2f}"
            draw_poly(canvas, det["poly"], colors["CLASS_MISMATCH"], label, thickness=2)
        else:
            label = f"pred {cname} {det['score']:.2f}"
            draw_poly(canvas, det["poly"], colors["TP_DET"], label, thickness=1)

    for gi, (gt_cls, gt_poly) in enumerate(gts):
        status, match_idx, iou = gt_status[gi]
        if status == "FN":
            label = f"FN gt {class_name(classes, gt_cls)}"
            draw_poly(canvas, gt_poly, colors["FN"], label, thickness=3)
        elif status == "CLASS_MISMATCH":
            det_cls = dets[match_idx]["cls"] if match_idx is not None else -1
            label = (
                f"GT {class_name(classes, gt_cls)} / pred "
                f"{class_name(classes, det_cls)} IoU {iou:.2f}"
            )
            draw_poly(canvas, gt_poly, colors["CLASS_MISMATCH"], label, thickness=3)
        else:
            pass
            # label = f"GT {class_name(classes, gt_cls)}"
            # draw_poly(canvas, gt_poly, colors["TP_GT"], label, thickness=2)

    summary = (
        f"FP {counts['fp']} | FN {counts['fn']} | class mismatch {counts['class_mismatch']} | "
        f"IoU >= {thresholds['iou']:.2f}, conf >= {thresholds['conf']:.2f}"
    )
    draw_label(canvas, summary, 8, 24, (35, 35, 35))
    return canvas


def inspect_model(name, model_path, module, args):
    model_path = resolve_model_path(module, model_path)
    if not os.path.isfile(model_path):
        print(f"[WARN] {name}: model not found at {model_path} - skipping")
        return None

    print(f"\n=== Inspecting {name} ({model_path}) ===")
    session = make_session(module, model_path, args.providers)
    inp = session.get_inputs()[0]
    input_size = int(inp.shape[2])
    input_name = inp.name
    classes = args.classes
    num_classes = len(classes)
    conf_thres = args.conf if args.conf is not None else module.CONFMAT_CONF
    iou_thres = args.iou if args.iou is not None else module.CONFMAT_IOU

    model_root = Path(args.output_root) / slugify(name)
    legacy_dir = model_root / "error_detections"
    model_dir = model_root / "error_inspection"
    for old_dir in (legacy_dir, model_dir):
        if old_dir.exists():
            shutil.rmtree(old_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / "error_report.csv"

    img_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in module.IMG_EXTS
    )
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    rows = []
    saved = 0
    total_fp = 0
    total_fn = 0
    total_mismatch = 0
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
            outputs[0], input_size, w, h, num_classes, module.CONF_THRESHOLD_INFER, meta
        )
        dets.sort(key=lambda d: -d["score"])

        high_dets, det_status, gt_status, counts, has_error = match_errors(
            module, dets, gts, num_classes, conf_thres, iou_thres
        )
        if has_error:
            rendered = render_error_image(
                img, classes, gts, high_dets, det_status, gt_status, counts,
                {"conf": conf_thres, "iou": iou_thres},
            )
            out_name = (
                f"{img_path.stem}__fp{counts['fp']}_fn{counts['fn']}"
                f"_mismatch{counts['class_mismatch']}.jpg"
            )
            cv2.imwrite(str(model_dir / out_name), rendered)
            fp_best_ious = [
                f"{iou:.4f}"
                for status, _match_idx, iou in det_status
                if status == "FP"
            ]
            rows.append([
                img_path.name,
                out_name,
                str(label_path),
                len(gts),
                counts["fp"],
                counts["fn"],
                counts["class_mismatch"],
                counts["tp"],
                ";".join(fp_best_ious),
                " | ".join(label_issues),
            ])
            saved += 1
            total_fp += counts["fp"]
            total_fn += counts["fn"]
            total_mismatch += counts["class_mismatch"]

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"  [{idx:>4}/{len(img_paths)}] saved {saved} error images")

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image", "annotated_image", "label_file", "gt_count",
            "false_positives", "false_negatives", "class_mismatches",
            "true_positives", "fp_best_ious", "label_issues",
        ])
        writer.writerows(rows)

    print(
        f"[OK] {name}: saved {saved} images to {model_dir} "
        f"(FP={total_fp}, FN={total_fn}, class mismatches={total_mismatch})"
    )
    return {
        "model": name,
        "saved": saved,
        "fp": total_fp,
        "fn": total_fn,
        "class_mismatch": total_mismatch,
        "output_dir": str(model_dir),
    }


def build_model_list(args):
    selected = []
    if args.kind in ("all", "fp32"):
        models = parse_model_args(args.fp32_model) if args.fp32_model else dict(YOLO_FP32.DEFAULT_MODELS)
        selected.extend((name, path, YOLO_FP32) for name, path in models.items())
    if args.kind in ("all", "int8"):
        models = parse_model_args(args.int8_model) if args.int8_model else dict(YOLO_INT8.DEFAULT_MODELS)
        selected.extend((name, path, YOLO_INT8) for name, path in models.items())
    return selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=YOLO_FP32.DEFAULT_VAL_IMAGES,
                        help="Directory of validation images.")
    parser.add_argument("--labels", default=YOLO_FP32.DEFAULT_VAL_LABELS,
                        help="Directory of YOLO-OBB label files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Benchmark root. Outputs go to <root>/<model>/error_inspection.")
    parser.add_argument("--kind", choices=["all", "fp32", "int8"], default="all",
                        help="Which default model set to inspect.")
    parser.add_argument("--fp32-model", action="append", default=None, metavar="NAME=PATH",
                        help="Custom FP32 model entry. Repeatable.")
    parser.add_argument("--int8-model", action="append", default=None, metavar="NAME=PATH",
                        help="Custom INT8 model entry. Repeatable.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime execution providers.")
    parser.add_argument("--classes", nargs="+", default=YOLO_FP32.DEFAULT_CLASSES,
                        help="Class names in order.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override false-positive matching confidence threshold.")
    parser.add_argument("--iou", type=float, default=None,
                        help="Override false-negative matching IoU threshold.")
    parser.add_argument("--ignore-empty-labels", action="store_true",
                        help="Skip images with no GT boxes.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional quick-test limit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    summaries = []
    for name, model_path, module in build_model_list(args):
        summary = inspect_model(name, model_path, module, args)
        if summary:
            summaries.append(summary)

    if not summaries:
        raise SystemExit("No models inspected.")

    print("\n=== Inspection summary ===")
    for row in summaries:
        print(
            f"{row['model']}: saved={row['saved']} "
            f"FP={row['fp']} FN={row['fn']} mismatch={row['class_mismatch']} "
            f"-> {row['output_dir']}"
        )


if __name__ == "__main__":
    main()
