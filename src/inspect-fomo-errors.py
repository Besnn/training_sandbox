#!/usr/bin/env python3
"""Render FOMO false positives/false negatives for visual inspection.

FOMO predicts centroids instead of boxes, so this script matches predictions to
ground-truth centroids by distance in output-grid cells. Any image with a false
positive, false negative, or class mismatch is annotated and written to:

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


def load_eval_module():
    spec = importlib.util.spec_from_file_location("evaluate_fomo", SCRIPT_DIR / "evaluate-fomo.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load evaluate-fomo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FOMO = load_eval_module()


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


def class_name(classes, cls_id):
    if 0 <= int(cls_id) < len(classes):
        return classes[int(cls_id)]
    return str(cls_id)


def centroid_distance(det, gt):
    dx = det["cell"][0] - gt[1][0]
    dy = det["cell"][1] - gt[1][1]
    return float((dx * dx + dy * dy) ** 0.5)


def match_errors(dets, gts, conf_thres, dist_thres):
    high_dets = [d for d in dets if d["score"] >= conf_thres]
    det_matches = [None] * len(high_dets)
    gt_matches = [None] * len(gts)

    candidate_matches = []
    for di, det in enumerate(high_dets):
        for gi, gt in enumerate(gts):
            dist = centroid_distance(det, gt)
            if dist <= dist_thres:
                candidate_matches.append((dist, di, gi))

    # Pair closest centroids first. This avoids a higher-confidence but farther
    # prediction claiming a GT that another prediction fits better.
    candidate_matches.sort(key=lambda item: item[0])
    used_dets = set()
    used_gts = set()
    for dist, di, gi in candidate_matches:
        if di in used_dets or gi in used_gts:
            continue
        det_matches[di] = (gi, dist)
        gt_matches[gi] = (di, dist)
        used_dets.add(di)
        used_gts.add(gi)

    det_status = []
    gt_status = []
    has_error = False

    for di, det in enumerate(high_dets):
        match = det_matches[di]
        if match is None:
            det_status.append(("FP", None, 0.0))
            has_error = True
            continue
        gi, dist = match
        gt_cls = gts[gi][0]
        if det["cls"] == gt_cls:
            det_status.append(("TP", gi, dist))
        else:
            det_status.append(("CLASS_MISMATCH", gi, dist))
            has_error = True

    for gi, (gt_cls, _gt_cell) in enumerate(gts):
        match = gt_matches[gi]
        if match is None:
            gt_status.append(("FN", None, 0.0))
            has_error = True
            continue
        di, dist = match
        det_cls = high_dets[di]["cls"]
        if det_cls == gt_cls:
            gt_status.append(("TP", di, dist))
        else:
            gt_status.append(("CLASS_MISMATCH", di, dist))

    counts = {
        "fp": sum(1 for status, _, _ in det_status if status == "FP"),
        "fn": sum(1 for status, _, _ in gt_status if status == "FN"),
        "class_mismatch": sum(1 for status, _, _ in det_status if status == "CLASS_MISMATCH"),
        "tp": sum(1 for status, _, _ in det_status if status == "TP"),
    }
    return high_dets, det_status, gt_status, counts, has_error


def cell_to_pixel(cell, img_w, img_h, grid_w, grid_h,
                  crop_left=None, crop_bottom=None):
    """Map a grid cell back to RAW image pixels, accounting for the input crop.

    The model sees `image[:img_h-crop_bottom, crop_left:]`, so the grid covers
    that cropped region. We map the cell into cropped pixels first, then shift
    by `crop_left` to land on the raw image.
    """
    if crop_left is None:
        crop_left = getattr(FOMO, "CROP_LEFT", 0)
    if crop_bottom is None:
        crop_bottom = getattr(FOMO, "CROP_BOTTOM", 0)
    cropped_w = max(1, img_w - crop_left)
    cropped_h = max(1, img_h - crop_bottom)
    x = crop_left + cell[0] / grid_w * cropped_w
    y = cell[1] / grid_h * cropped_h
    return int(round(x)), int(round(y))


def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    label_w = tw + 4
    label_h = th + baseline + 4

    candidates = [
        (x + 8, y - 8),
        (x + 8, y + label_h + 12),
        (x - label_w - 8, y - 8),
        (x - label_w - 8, y + label_h + 12),
    ]
    for cx, cy in candidates:
        left = cx
        right = cx + label_w
        top = cy - label_h
        bottom = cy
        if left >= 0 and top >= 0 and right < img.shape[1] and bottom < img.shape[0]:
            break
    else:
        cx = max(0, min(x + 8, img.shape[1] - label_w))
        cy = max(label_h, min(y - 8, img.shape[0] - 4))

    cv2.rectangle(img, (cx, cy - label_h), (cx + label_w, cy), color, -1)
    cv2.putText(img, text, (cx + 2, cy - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_centroid(img, x, y, color, label, radius=8, thickness=2):
    cv2.circle(img, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x - radius - 3, y), (x + radius + 3, y), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, (x, y - radius - 3), (x, y + radius + 3), color, thickness, lineType=cv2.LINE_AA)
    draw_label(img, label, x, y, color)


def render_error_image(img, classes, gts, dets, det_status, gt_status, counts, grid_size, thresholds):
    canvas = img.copy()
    img_h, img_w = canvas.shape[:2]
    grid_w, grid_h = grid_size

    colors = {
        "TP_GT": (85, 150, 85),
        "TP_DET": (150, 170, 80),
        "FP": (30, 30, 230),
        "FN": (0, 145, 255),
        "CLASS_MISMATCH": (0, 210, 255),
    }

    for gi, (gt_cls, gt_cell) in enumerate(gts):
        x, y = cell_to_pixel(gt_cell, img_w, img_h, grid_w, grid_h)
        status, match_idx, dist = gt_status[gi]
        if status == "FN":
            label = f"FN gt {class_name(classes, gt_cls)}"
            draw_centroid(canvas, x, y, colors["FN"], label, radius=10, thickness=3)
        elif status == "CLASS_MISMATCH":
            det_cls = dets[match_idx]["cls"] if match_idx is not None else -1
            label = f"GT {class_name(classes, gt_cls)} / pred {class_name(classes, det_cls)} d {dist:.1f}"
            draw_centroid(canvas, x, y, colors["CLASS_MISMATCH"], label, radius=10, thickness=3)
        else:
            label = f"GT {class_name(classes, gt_cls)}"
            draw_centroid(canvas, x, y, colors["TP_GT"], label, radius=7, thickness=1)

    for di, det in enumerate(dets):
        x, y = cell_to_pixel(det["cell"], img_w, img_h, grid_w, grid_h)
        status, match_idx, dist = det_status[di]
        cname = class_name(classes, det["cls"])
        if status == "FP":
            label = f"FP pred {cname} {det['score']:.2f}"
            draw_centroid(canvas, x, y, colors["FP"], label, radius=10, thickness=3)
        elif status == "CLASS_MISMATCH":
            label = f"Wrong pred {cname} {det['score']:.2f} d {dist:.1f}"
            draw_centroid(canvas, x, y, colors["CLASS_MISMATCH"], label, radius=9, thickness=2)
        else:
            label = f"pred {cname} {det['score']:.2f}"
            draw_centroid(canvas, x, y, colors["TP_DET"], label, radius=7, thickness=1)

    summary = (
        f"FP {counts['fp']} | FN {counts['fn']} | class mismatch {counts['class_mismatch']} | "
        f"dist <= {thresholds['dist']:.1f} grid cells, conf >= {thresholds['conf']:.2f}"
    )
    draw_label(canvas, summary, 8, 24, (35, 35, 35))
    return canvas


def inspect_model(name, model_path, args):
    model_path = FOMO.resolve_model_path(model_path)
    if not os.path.isfile(model_path):
        print(f"[WARN] {name}: model not found at {model_path} - skipping")
        return None

    print(f"\n=== Inspecting {name} ({model_path}) ===")
    session = ort.InferenceSession(model_path, providers=args.providers)
    inp = session.get_inputs()[0]
    input_name = inp.name
    try:
        input_size = int(inp.shape[2])
    except (TypeError, ValueError):
        input_size = 480
    grid_w, grid_h = FOMO.infer_grid_size(session, input_size, input_name)

    classes = args.classes
    num_classes = len(classes)
    conf_thres = args.conf if args.conf is not None else FOMO.CONFMAT_CONF
    dist_thres = args.dist if args.dist is not None else FOMO.CONFMAT_DIST

    model_root = Path(args.output_root) / slugify(name)
    legacy_dir = model_root / "error_detections"
    model_dir = model_root / "error_inspection"
    for old_dir in (legacy_dir, model_dir):
        if old_dir.exists():
            shutil.rmtree(old_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / "error_report.csv"

    img_paths = sorted(
        p for p in Path(args.images).iterdir() if p.suffix.lower() in FOMO.IMG_EXTS
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
        raw_h, raw_w = img.shape[:2]
        gts = FOMO.load_gt(
            str(Path(args.labels) / f"{img_path.stem}.txt"),
            grid_w, grid_h,
            raw_w=raw_w, raw_h=raw_h,
        )
        if args.ignore_empty_labels and not gts:
            continue

        blob = FOMO.preprocess(img, input_size)
        outputs = session.run(None, {input_name: blob})
        dets = FOMO.postprocess(outputs[0], num_classes, FOMO.CONF_THRESHOLD_INFER)
        dets.sort(key=lambda d: -d["score"])

        high_dets, det_status, gt_status, counts, has_error = match_errors(
            dets, gts, conf_thres, dist_thres
        )
        if has_error:
            rendered = render_error_image(
                img, classes, gts, high_dets, det_status, gt_status, counts,
                (grid_w, grid_h), {"conf": conf_thres, "dist": dist_thres},
            )
            out_name = (
                f"{img_path.stem}__fp{counts['fp']}_fn{counts['fn']}"
                f"_mismatch{counts['class_mismatch']}.jpg"
            )
            cv2.imwrite(str(model_dir / out_name), rendered)
            rows.append([
                img_path.name,
                out_name,
                counts["fp"],
                counts["fn"],
                counts["class_mismatch"],
                counts["tp"],
            ])
            saved += 1
            total_fp += counts["fp"]
            total_fn += counts["fn"]
            total_mismatch += counts["class_mismatch"]

        if idx % 10 == 0 or idx == len(img_paths):
            print(f"  [{idx:>4}/{len(img_paths)}] saved {saved} error images")

    with open(report_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "annotated_image", "false_positives", "false_negatives", "class_mismatches", "true_positives"])
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=FOMO.DEFAULT_VAL_IMAGES,
                        help="Directory of validation images.")
    parser.add_argument("--labels", default=FOMO.DEFAULT_VAL_LABELS,
                        help="Directory of YOLO-centroid label files.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT,
                        help="Benchmark root. Outputs go to <root>/<model>/error_inspection.")
    parser.add_argument("--model", action="append", default=None, metavar="NAME=PATH",
                        help="Custom model entry. Repeatable. Defaults to evaluate-fomo.py DEFAULT_MODELS.")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"],
                        help="ONNX Runtime execution providers.")
    parser.add_argument("--classes", nargs="+", default=FOMO.DEFAULT_CLASSES,
                        help="Class names in order.")
    parser.add_argument("--conf", type=float, default=None,
                        help="Override false-positive matching confidence threshold.")
    parser.add_argument("--dist", type=float, default=None,
                        help="Override false-negative matching distance threshold in grid cells.")
    parser.add_argument("--ignore-empty-labels", action="store_true",
                        help="Skip images with no GT centroids.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional quick-test limit.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.images):
        raise SystemExit(f"Images directory not found: {args.images}")
    if not os.path.isdir(args.labels):
        raise SystemExit(f"Labels directory not found: {args.labels}")

    models = parse_model_args(args.model) if args.model else dict(FOMO.DEFAULT_MODELS)
    summaries = []
    for name, model_path in models.items():
        summary = inspect_model(name, model_path, args)
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
