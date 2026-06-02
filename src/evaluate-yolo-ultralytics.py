#!/usr/bin/env python3
"""Evaluate YOLO-OBB models via Ultralytics and produce a results table.

Runs one or more YOLO models (ONNX, PT, etc.) on a validation image folder,
computes per-conf Detections/TP/FP/FN/Precision/Recall/F1 and model-level
mAP50/mAP50-95, then writes CSV, Markdown, and LaTeX tables in the same
format as make-results-table.py.

Usage:
    python3 evaluate-yolo-ultralytics.py \\
        --model "YOLOv8n-FP32=models/yolov8n/yolov8n-obb-fp32.onnx" \\
        --model "YOLOv8n-INT8=models/yolov8n/yolov8n-obb-int8.onnx" \\
        --images datasets/yolo_pl_test/images \\
        --labels datasets/yolo_pl_test/labels \\
        --conf 0.25 0.5 0.75 \\
        --output results-table
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGES = SCRIPT_DIR / "datasets/yolo_pl_test/images"
DEFAULT_LABELS = SCRIPT_DIR / "datasets/yolo_pl_test/labels"
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
DEFAULT_CONF_THRESHOLDS = [0.25, 0.5, 0.75]
DEFAULT_GROUP = "YOLO"

CONF_INFER = 0.001   # keep all boxes for mAP computation
NMS_IOU = 0.1        # tight NMS so duplicate boxes don't inflate TP
MATCH_IOU = 0.25     # IoU threshold for TP/FP/FN matching
IOU_SWEEP = np.linspace(0.5, 0.95, 10)   # COCO-style sweep for mAP50-95


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def poly_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
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
def load_gt(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    gts = []
    if not label_path.is_file():
        return gts
    for raw in label_path.read_text().splitlines():
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
        gts.append((cls, xy.astype(np.float32)))
    return gts


# --------------------------------------------------------------------------- #
# Inference conversion
# --------------------------------------------------------------------------- #
def obb_to_dets(result) -> list[dict]:
    obb = result.obb
    if obb is None or len(obb) == 0:
        return []
    polys = obb.xyxyxyxy.cpu().numpy()
    confs = obb.conf.cpu().numpy()
    classes = obb.cls.cpu().numpy()
    return [
        {"cls": int(classes[i]), "score": float(confs[i]), "poly": polys[i]}
        for i in range(len(confs))
    ]


# --------------------------------------------------------------------------- #
# mAP computation
# --------------------------------------------------------------------------- #
def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """101-point interpolated AP (COCO-style)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    return float(np.trapz(np.interp(x, mrec, mpre), x))


def match_for_map(dets: list[dict], gts: list[tuple]) -> np.ndarray:
    """Return (N_det, N_iou_thresholds) bool TP array. Dets must be score-sorted."""
    niou = len(IOU_SWEEP)
    tp = np.zeros((len(dets), niou), dtype=bool)
    if not dets or not gts:
        return tp
    iou_mat = np.zeros((len(dets), len(gts)), dtype=np.float32)
    for di, d in enumerate(dets):
        for gi, (gc, gp) in enumerate(gts):
            if d["cls"] == gc:
                iou_mat[di, gi] = poly_iou(d["poly"], gp)
    for ti, thr in enumerate(IOU_SWEEP):
        used = np.zeros(len(gts), dtype=bool)
        for di in range(len(dets)):
            best_iou, best_gi = thr, -1
            for gi in range(len(gts)):
                if not used[gi] and iou_mat[di, gi] >= best_iou:
                    best_iou = iou_mat[di, gi]
                    best_gi = gi
            if best_gi >= 0:
                tp[di, ti] = True
                used[best_gi] = True
    return tp


def compute_map(
    all_dets: list[list[dict]],
    all_gts: list[list[tuple]],
    num_classes: int,
) -> tuple[float, float]:
    """Compute mAP50 and mAP50-95 across all images."""
    niou = len(IOU_SWEEP)
    ap = np.zeros((num_classes, niou), dtype=np.float64)

    for cls in range(num_classes):
        cls_scores, cls_tp, n_gt = [], [], 0
        for dets, gts in zip(all_dets, all_gts):
            cls_dets = sorted(
                [d for d in dets if d["cls"] == cls],
                key=lambda d: -d["score"],
            )
            cls_gts = [g for g in gts if g[0] == cls]
            n_gt += len(cls_gts)
            tp_mat = match_for_map(cls_dets, cls_gts)
            cls_scores.extend(d["score"] for d in cls_dets)
            for ti in range(niou):
                cls_tp.append(tp_mat[:, ti] if len(cls_dets) else np.array([], dtype=bool))

        if n_gt == 0 or not cls_scores:
            continue

        order = np.argsort(cls_scores)[::-1]
        for ti in range(niou):
            tp_arr = np.concatenate([cls_tp[i * niou + ti] for i in range(len(all_dets))])
            tp_arr = tp_arr[order]
            cum_tp = np.cumsum(tp_arr)
            cum_fp = np.cumsum(~tp_arr)
            precision = cum_tp / (cum_tp + cum_fp + 1e-9)
            recall = cum_tp / (n_gt + 1e-9)
            ap[cls, ti] = compute_ap(recall, precision)

    return float(ap[:, 0].mean()), float(ap.mean())


# --------------------------------------------------------------------------- #
# TP/FP/FN counting at a confidence threshold
# --------------------------------------------------------------------------- #
def count_errors(
    dets: list[dict],
    gts: list[tuple],
    conf_thres: float,
    iou_thres: float,
) -> tuple[int, int, int, int]:
    """Return (detections, tp, fp, fn) for one image."""
    high = [d for d in dets if d["score"] >= conf_thres]
    high.sort(key=lambda d: -d["score"])
    det_matched = [False] * len(high)
    gt_matched = [False] * len(gts)

    candidates = []
    for di, det in enumerate(high):
        for gi, (_, gp) in enumerate(gts):
            iou = poly_iou(det["poly"], gp)
            if iou >= iou_thres:
                candidates.append((iou, di, gi))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_d, used_g = set(), set()
    for _, di, gi in candidates:
        if di not in used_d and gi not in used_g:
            det_matched[di] = True
            gt_matched[gi] = True
            used_d.add(di)
            used_g.add(gi)

    tp = sum(det_matched)
    fp = len(high) - tp
    fn = len(gts) - sum(gt_matched)
    return len(high), tp, fp, fn


# --------------------------------------------------------------------------- #
# Table row
# --------------------------------------------------------------------------- #
@dataclass
class TableRow:
    group: str
    configuration: str
    confidence: float
    detections: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    map50: float
    map5095: float


# --------------------------------------------------------------------------- #
# Table writers (same format as make-results-table.py)
# --------------------------------------------------------------------------- #
def latex_escape(s: str) -> str:
    for ch, rep in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                    ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                    ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(ch, rep)
    return s


def write_csv(path: Path, rows: list[TableRow], decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Configuration", "Detections", "TP", "FP", "FN",
                    "Precision", "Recall", "F1", "mAP50", "mAP50-95"])
        for r in rows:
            w.writerow([r.group, r.configuration, r.detections, r.tp, r.fp, r.fn,
                        f"{r.precision:.{decimals}f}", f"{r.recall:.{decimals}f}",
                        f"{r.f1:.{decimals}f}", f"{r.map50:.{decimals}f}",
                        f"{r.map5095:.{decimals}f}"])


def write_markdown(path: Path, rows: list[TableRow], decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current_group = None
    with path.open("w") as f:
        f.write("| Configuration | Detections | TP | FP | FN | Precision | Recall | F1 | mAP50 | mAP50-95 |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for r in rows:
            if r.group != current_group:
                f.write(f"| **{r.group}** | | | | | | | | | |\n")
                current_group = r.group
            f.write(f"| {r.configuration} | {r.detections} | {r.tp} | {r.fp} | {r.fn} | "
                    f"{r.precision:.{decimals}f} | {r.recall:.{decimals}f} | {r.f1:.{decimals}f} | "
                    f"{r.map50:.{decimals}f} | {r.map5095:.{decimals}f} |\n")


def write_latex(path: Path, rows: list[TableRow], decimals: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    current_group = None
    with path.open("w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\setlength{\\tabcolsep}{5pt}\n\\renewcommand{\\arraystretch}{1.08}\n")
        f.write("\\begin{tabular}{lrrrrrrrrr}\n\\hline\n")
        for r in rows:
            if r.group != current_group:
                f.write(f"\\multicolumn{{10}}{{c}}{{\\textbf{{{latex_escape(r.group)}}}}} \\\\\n\\hline\n")
                f.write("\\textbf{Configuration} & \\textit{Detections} & \\textit{TP} & "
                        "\\textit{FP} & \\textit{FN} & \\textit{Precision} & \\textit{Recall} & "
                        "\\textit{F1} & \\textit{mAP50} & \\textit{mAP50--95} \\\\\n\\hline\n")
                current_group = r.group
            cfg = latex_escape(r.configuration)
            f.write(f"{cfg} & {r.detections} & {r.tp} & {r.fp} & {r.fn} & "
                    f"{r.precision:.{decimals}f} & {r.recall:.{decimals}f} & {r.f1:.{decimals}f} & "
                    f"{r.map50:.{decimals}f} & {r.map5095:.{decimals}f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Detection performance comparison.}\n")
        f.write("\\label{tab:detection-results}\n\\end{table}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="NAME=PATH",
                        help="Model to evaluate. Repeatable.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--conf", nargs="+", type=float, default=DEFAULT_CONF_THRESHOLDS,
                        help="Confidence threshold(s) for Detections/TP/FP/FN rows.")
    parser.add_argument("--iou", type=float, default=MATCH_IOU,
                        help="IoU threshold for TP/FP/FN matching.")
    parser.add_argument("--group", default=DEFAULT_GROUP,
                        help="Section label in the table.")
    parser.add_argument("--output", type=Path, default=Path("results-table"),
                        help="Output path prefix (writes .csv, .md, .tex).")
    parser.add_argument("--format", choices=("all", "csv", "md", "tex"), default="all")
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def parse_model(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise SystemExit(f"--model expects NAME=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    return name.strip(), path.strip()


def resolve_model_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() or p.is_file():
        return p
    candidate = SCRIPT_DIR / p
    if candidate.is_file():
        return candidate
    return p


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        raise SystemExit("ultralytics is not installed.")

    if not args.images.is_dir():
        raise SystemExit(f"Images directory not found: {args.images}")
    if not args.labels.is_dir():
        raise SystemExit(f"Labels directory not found: {args.labels}")

    img_paths = sorted(
        p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if args.max_images:
        img_paths = img_paths[:args.max_images]

    all_rows: list[TableRow] = []

    for model_raw in args.model:
        name, path_raw = parse_model(model_raw)
        model_path = resolve_model_path(path_raw)
        if not model_path.is_file():
            raise SystemExit(f"Model not found: {model_path}")

        print(f"\n=== {name}: {model_path} ===")
        model = YOLO(str(model_path))

        all_dets: list[list[dict]] = []
        all_gts: list[list[tuple]] = []

        for idx, img_path in enumerate(img_paths, start=1):
            result = model(str(img_path), conf=CONF_INFER, iou=NMS_IOU, verbose=False)[0]
            h, w = result.orig_shape
            dets = obb_to_dets(result)
            dets.sort(key=lambda d: -d["score"])
            gts = load_gt(args.labels / f"{img_path.stem}.txt", w, h)
            all_dets.append(dets)
            all_gts.append(gts)
            if idx % 50 == 0 or idx == len(img_paths):
                print(f"  [{idx:>4}/{len(img_paths)}]")

        print("  Computing mAP...")
        map50, map5095 = compute_map(all_dets, all_gts, len(args.classes))
        print(f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")

        for conf in sorted(args.conf):
            total_det = total_tp = total_fp = total_fn = 0
            for dets, gts in zip(all_dets, all_gts):
                det, tp, fp, fn = count_errors(dets, gts, conf, args.iou)
                total_det += det
                total_tp += tp
                total_fp += fp
                total_fn += fn
            precision = total_tp / total_det if total_det else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            all_rows.append(TableRow(
                group=args.group,
                configuration=f"{name}@{conf}",
                confidence=conf,
                detections=total_det,
                tp=total_tp,
                fp=total_fp,
                fn=total_fn,
                precision=precision,
                recall=recall,
                f1=f1,
                map50=map50,
                map5095=map5095,
            ))
            print(f"  @{conf}: det={total_det} tp={total_tp} fp={total_fp} fn={total_fn} "
                  f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    if not all_rows:
        raise SystemExit("No results to write.")

    suffix = args.output.suffix.lower()
    formats = [suffix[1:]] if suffix in {".csv", ".md", ".tex"} else (
        ["csv", "md", "tex"] if args.format == "all" else [args.format]
    )
    base = args.output if not suffix else args.output.with_suffix("")

    if "csv" in formats:
        write_csv(base.with_suffix(".csv"), all_rows, args.precision)
        print(f"\n[OK] {base.with_suffix('.csv')}")
    if "md" in formats:
        write_markdown(base.with_suffix(".md"), all_rows, args.precision)
        print(f"[OK] {base.with_suffix('.md')}")
    if "tex" in formats:
        write_latex(base.with_suffix(".tex"), all_rows, args.precision)
        print(f"[OK] {base.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
