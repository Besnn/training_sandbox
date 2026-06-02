#!/usr/bin/env python3
"""Run all local FOMO and YOLO OBB evaluation scripts.

This script is only an orchestrator: it delegates model evaluation to:
  - evaluate-fomo.py
  - evaluate-yolo-obb.py
  - evaluate-yolo-obb-int8.py

After the evaluators finish, it collects their per-model metrics.csv files into
one compact CSV and Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_IMAGES_DIR = SCRIPT_DIR / "datasets/yolo_pl_test/images"
DEFAULT_FOMO_LABELS_DIR = SCRIPT_DIR / "datasets/yolo_pl_test_centroid/labels"
DEFAULT_YOLO_LABELS_DIR = SCRIPT_DIR / "datasets/yolo_pl_test/labels"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "benchmark_results/all-model-eval"

FOMO_MODELS = [
    # ("FOMO-FP32", "models/fomo/fomo-480.onnx"),
    # ("FOMO-INT8", "models/fomo/fomo-480-int8.onnx"),
]

YOLO_MODELS = [
    # ("YOLOv8n-OBB-FP32", "models/yolov8n/yolov8n-obb-fp32.onnx"),
    # ("YOLOv8n-OBB-FP16", "models/yolov8n/yolov8n-obb-fp16.onnx"),
    ("YOLOv8m-OBB-FP32", "models/yolov8m-obb-onnx/yolov8m-obb-fp32.onnx"),
    # ("YOLO26n-OBB-FP32", "models/yolo26n/yolo26n-obb-fp32.onnx"),
    # ("YOLO26n-OBB-FP16", "models/yolo26n/yolo26n-obb-fp16.onnx"),
]

YOLO_INT8_MODELS = [
    # ("YOLOv8n-OBB-INT8", "models/yolov8n/yolov8n-obb-int8.onnx"),
    # ("YOLO26n-OBB-INT8", "models/yolo26n/yolo26n-obb-int8.onnx"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="Validation images directory.")
    parser.add_argument("--fomo-labels", type=Path, default=DEFAULT_FOMO_LABELS_DIR,
                        help="Centroid labels directory for FOMO evaluation.")
    parser.add_argument("--yolo-labels", type=Path, default=DEFAULT_YOLO_LABELS_DIR,
                        help="OBB labels directory for YOLO evaluation.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Root directory for evaluator outputs and summary files.")
    parser.add_argument("--provider", action="append", dest="providers",
                        default=None,
                        help="ONNX Runtime execution provider. Repeat for multiple providers.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used to run evaluator scripts.")
    parser.add_argument("--ignore-empty-labels", action="store_true",
                        help="Pass --ignore-empty-labels to all evaluators.")
    parser.add_argument("--conf-threshold", action="append", type=float,
                        dest="conf_thresholds", default=None,
                        help="Confusion-matrix confidence threshold. Repeat for "
                             "multiple table rows, e.g. 0.25, 0.5, 0.75.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print evaluator commands without running them.")
    return parser.parse_args()


def existing_model_args(models: list[tuple[str, str]]) -> list[str]:
    args: list[str] = []
    for name, relative_path in models:
        model_path = SCRIPT_DIR / relative_path
        if model_path.is_file():
            args.extend(["--model", f"{name}={relative_path}"])
        else:
            print(f"[WARN] Skipping missing model: {relative_path}", file=sys.stderr)
    return args


def run_command(command: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(sh_quote(part) for part in command))
    if not dry_run:
        subprocess.run(command, check=True)


def sh_quote(value: str) -> str:
    if value and all(ch.isalnum() or ch in "/._-:=+" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def evaluator_command(
    python_bin: str,
    script_name: str,
    images: Path,
    labels: Path,
    output_dir: Path,
    providers: list[str],
    model_args: list[str],
    ignore_empty_labels: bool,
    conf_threshold: float,
) -> list[str]:
    command = [
        python_bin,
        str(SCRIPT_DIR / script_name),
        "--images",
        str(images),
        "--labels",
        str(labels),
        "--output-dir",
        str(output_dir),
        "--providers",
        *providers,
        "--confmat-conf",
        f"{conf_threshold:g}",
        *model_args,
    ]
    if ignore_empty_labels:
        command.append("--ignore-empty-labels")
    return command


def collect_summary(output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metrics_path in sorted(output_dir.rglob("metrics.csv")):
        parts = metrics_path.relative_to(output_dir).parts
        family = "/".join(parts[:-2]) if len(parts) > 2 else parts[0]
        model_name = metrics_path.parent.name

        with metrics_path.open(newline="") as f:
            csv_rows = list(csv.reader(f))

        if csv_rows and csv_rows[0][:1] == ["model"] and len(csv_rows[0]) > 1:
            model_name = csv_rows[0][1]

        header = None
        mean = None
        for row in csv_rows:
            if row and row[0] == "class":
                header = row
            elif row and row[0] == "mean":
                mean = row
                break

        if not header or not mean:
            continue

        values = dict(zip(header, mean))
        fomo_ap_header = next((h for h in header if h.startswith("AP@d=")), "")
        rows.append({
            "family": family,
            "model": model_name,
            "GT": values.get("GT", ""),
            "precision": values.get("precision", ""),
            "recall": values.get("recall", ""),
            "F1": values.get("F1", ""),
            "AP_primary": values.get("AP50", values.get(fomo_ap_header, "")),
            "AP_sweep": values.get("AP50-95", values.get("AP@d=sweep_mean", "")),
        })
    return rows


def write_summary(output_dir: Path, rows: list[dict[str, str]]) -> None:
    summary_csv = output_dir / "all-models-summary.csv"
    summary_md = output_dir / "all-models-summary.md"
    fieldnames = [
        "family",
        "model",
        "GT",
        "precision",
        "recall",
        "F1",
        "AP_primary",
        "AP_sweep",
    ]

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_md.open("w") as f:
        f.write("| Family | Configuration | GT | Precision | Recall | F1 | AP primary | AP sweep |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['family']} | {row['model']} | {row['GT']} | "
                f"{row['precision']} | {row['recall']} | {row['F1']} | "
                f"{row['AP_primary']} | {row['AP_sweep']} |\n"
            )

    print(f"[OK] Summary written to {summary_csv}")
    print(f"[OK] Markdown table written to {summary_md}")


def main() -> None:
    args = parse_args()
    providers = args.providers or ["CPUExecutionProvider"]
    conf_thresholds = args.conf_thresholds or [0.25, 0.5, 0.75]

    fomo_args = existing_model_args(FOMO_MODELS)
    yolo_args = existing_model_args(YOLO_MODELS)
    yolo_int8_args = existing_model_args(YOLO_INT8_MODELS)

    commands = []
    for threshold in conf_thresholds:
        threshold_dir = args.output_dir / f"conf-{threshold:g}"
        if fomo_args:
            commands.append(evaluator_command(
                args.python,
                "evaluate-fomo.py",
                args.images,
                args.fomo_labels,
                threshold_dir / "fomo",
                providers,
                fomo_args,
                args.ignore_empty_labels,
                threshold,
            ))
        else:
            print("[WARN] No FOMO models found.", file=sys.stderr)

        if yolo_args:
            commands.append(evaluator_command(
                args.python,
                "evaluate-yolo-obb.py",
                args.images,
                args.yolo_labels,
                threshold_dir / "yolo",
                providers,
                yolo_args,
                args.ignore_empty_labels,
                threshold,
            ))
        else:
            print("[WARN] No FP32/FP16 YOLO OBB models found.", file=sys.stderr)

        if yolo_int8_args:
            commands.append(evaluator_command(
                args.python,
                "evaluate-yolo-obb-int8.py",
                args.images,
                args.yolo_labels,
                threshold_dir / "yolo-int8",
                providers,
                yolo_int8_args,
                args.ignore_empty_labels,
                threshold,
            ))
        else:
            print("[WARN] No INT8 YOLO OBB models found.", file=sys.stderr)

    if args.dry_run:
        for command in commands:
            run_command(command, dry_run=True)
        print("[DRY-RUN] Summary aggregation skipped.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for command in commands:
        run_command(command, dry_run=False)

    write_summary(args.output_dir, collect_summary(args.output_dir))


if __name__ == "__main__":
    main()
