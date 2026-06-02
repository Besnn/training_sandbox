#!/usr/bin/env python3
"""Serve a local web UI for benchmark error inspection artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_RESULTS_ROOT = BENCHMARKS_DIR / "benchmark_results/inspection_folders"
DEFAULT_IMAGES_DIR = BENCHMARKS_DIR / "datasets/yolo_pl_test/images"
DEFAULT_LABELS_DIR = BENCHMARKS_DIR / "datasets/yolo_pl_test/labels"
DEFAULT_CENTROID_LABELS_DIR = BENCHMARKS_DIR / "datasets/yolo_pl_test_centroid/labels"
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
ERROR_IMAGE_RE = re.compile(
    r"^(?P<stem>.+)__fp(?P<fp>\d+)_fn(?P<fn>\d+)_mismatch(?P<mismatch>\d+)\.(?P<ext>jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help="Directory to scan for */inspection.json or legacy */error_inspection/error_report.csv.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="Directory containing original validation images.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS_DIR,
                        help="Directory containing YOLO OBB labels.")
    parser.add_argument("--centroid-labels", type=Path, default=DEFAULT_CENTROID_LABELS_DIR,
                        help="Directory containing YOLO centroid labels.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765,
                        help="Port to bind.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                        help="Class names in label order.")
    return parser.parse_args()


class AppState:
    def __init__(self, args: argparse.Namespace):
        self.results_root = args.results_root.resolve()
        self.images_dir = args.images.resolve()
        self.labels_dir = args.labels.resolve()
        self.centroid_labels_dir = args.centroid_labels.resolve()
        self.classes = args.classes

    def model_dirs(self) -> list[Path]:
        dirs = {path.parent for path in self.results_root.rglob("inspection.json")}
        dirs.update(path.parent for path in self.results_root.rglob("error_inspection") if path.is_dir())
        return sorted(dirs)

    def model_id(self, model_dir: Path) -> str:
        return model_dir.resolve().relative_to(self.results_root).as_posix()

    def model_dir(self, model_id: str) -> Path:
        path = (self.results_root / model_id).resolve()
        ensure_under(path, self.results_root)
        return path


def ensure_under(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise FileNotFoundError(path) from exc


def read_report(report_path: Path) -> list[dict[str, str]]:
    inspection_dir = report_path.parent
    if report_path.is_file():
        with report_path.open(newline="") as f:
            return list(csv.DictReader(f))

    rows = []
    for image_path in sorted(inspection_dir.iterdir()):
        match = ERROR_IMAGE_RE.match(image_path.name)
        if not match:
            continue
        raw_name = match.group("stem") + image_path.suffix
        rows.append({
            "image": raw_name,
            "annotated_image": image_path.name,
            "label_file": "",
            "gt_count": "",
            "false_positives": match.group("fp"),
            "false_negatives": match.group("fn"),
            "class_mismatches": match.group("mismatch"),
            "true_positives": "",
            "fp_best_ious": "",
            "label_issues": "",
        })
    return rows


def read_inspection(model_dir: Path) -> dict[str, object] | None:
    path = model_dir / "inspection.json"
    if not path.is_file():
        return None
    with path.open() as f:
        return json.load(f)


def int_field(row: dict[str, str], name: str) -> int:
    try:
        return int(float(row.get(name) or 0))
    except ValueError:
        return 0


def infer_architecture(records: list[dict[str, object]]) -> str:
    for record in records or []:
        for layer in (record.get("detections") or []) + (record.get("groundTruth") or []):
            if "centroid" in layer:
                return "fomo"
            if "polygon" in layer:
                return "yolo-obb"
    return "yolo-obb"


def model_summary(state: AppState, model_dir: Path) -> dict[str, object]:
    inspection = read_inspection(model_dir)
    if inspection is not None:
        totals_raw = inspection.get("totals", {})
        totals = {
            "images": int(totals_raw.get("processed", len(inspection.get("records", []))) or 0),
            "fp": int(totals_raw.get("fp", 0) or 0),
            "fn": int(totals_raw.get("fn", 0) or 0),
            "mismatch": int(totals_raw.get("mismatch", 0) or 0),
            "tp": int(totals_raw.get("tp", 0) or 0),
            "labelIssues": sum(
                1 for row in inspection.get("records", [])
                if row.get("labelIssues")
            ),
        }
        model = inspection.get("model", {})
        architecture = model.get("architecture") or infer_architecture(inspection.get("records", []))
        return {
            "id": state.model_id(model_dir),
            "name": model.get("name") or model_dir.name,
            "group": model_dir.parent.relative_to(state.results_root).as_posix()
            if model_dir.parent != state.results_root else "",
            "report": str(model_dir / "inspection.json"),
            "format": "inspection_json",
            "architecture": architecture,
            "gridWidth": model.get("gridWidth"),
            "gridHeight": model.get("gridHeight"),
            "totals": totals,
        }

    report_path = model_dir / "error_inspection/error_report.csv"
    rows = read_report(report_path)
    totals = {
        "images": len(rows),
        "fp": sum(int_field(row, "false_positives") for row in rows),
        "fn": sum(int_field(row, "false_negatives") for row in rows),
        "mismatch": sum(int_field(row, "class_mismatches") for row in rows),
        "tp": sum(int_field(row, "true_positives") for row in rows),
        "labelIssues": sum(1 for row in rows if (row.get("label_issues") or "").strip()),
    }
    return {
        "id": state.model_id(model_dir),
        "name": model_dir.name,
        "group": model_dir.parent.relative_to(state.results_root).as_posix()
        if model_dir.parent != state.results_root else "",
        "report": str(report_path),
        "format": "legacy_csv",
        "architecture": "yolo-obb",
        "gridWidth": None,
        "gridHeight": None,
        "totals": totals,
    }


def normalize_row(row: dict[str, str], index: int) -> dict[str, object]:
    return {
        "index": index,
        "image": row.get("image", ""),
        "annotatedImage": row.get("annotated_image", ""),
        "labelFile": row.get("label_file", ""),
        "gtCount": int_field(row, "gt_count"),
        "fp": int_field(row, "false_positives"),
        "fn": int_field(row, "false_negatives"),
        "mismatch": int_field(row, "class_mismatches"),
        "tp": int_field(row, "true_positives"),
        "fpBestIous": row.get("fp_best_ious", ""),
        "labelIssues": row.get("label_issues", ""),
    }


def normalize_inspection_record(record: dict[str, object], index: int) -> dict[str, object]:
    counts = record.get("counts", {}) or {}
    label_issues = record.get("labelIssues", []) or []
    fp_best_ious = []
    fp_best_distances = []
    for det in record.get("detections", []) or []:
        if det.get("status") != "FP":
            continue
        if "bestIoU" in det or "polygon" in det:
            fp_best_ious.append(f"{float(det.get('bestIoU', det.get('matchScore', 0.0))):.4f}")
        if "bestDistance" in det or "centroid" in det:
            fp_best_distances.append(f"{float(det.get('bestDistance', det.get('matchDistance', 0.0))):.2f}")

    return {
        "index": index,
        "image": record.get("image", ""),
        "rawImage": record.get("rawImage", ""),
        "annotatedImage": record.get("annotatedImage", ""),
        "labelFile": "",
        "gtCount": len(record.get("groundTruth", []) or []),
        "fp": int(counts.get("fp", 0) or 0),
        "fn": int(counts.get("fn", 0) or 0),
        "mismatch": int(counts.get("class_mismatch", counts.get("mismatch", 0)) or 0),
        "tp": int(counts.get("tp", 0) or 0),
        "fpBestIous": ";".join(fp_best_ious),
        "fpBestDistances": ";".join(fp_best_distances),
        "labelIssues": " | ".join(str(item) for item in label_issues),
        "hasError": bool(record.get("hasError", False)),
        "groundTruth": record.get("groundTruth", []),
        "detections": record.get("detections", []),
    }


def parse_label_file(path: Path, classes: list[str]) -> list[dict[str, object]]:
    labels = []
    if not path.is_file():
        return labels

    with path.open() as f:
        for line_no, raw in enumerate(f, start=1):
            parts = raw.strip().split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue

            label = {
                "line": line_no,
                "classId": cls,
                "className": classes[cls] if 0 <= cls < len(classes) else str(cls),
            }
            if len(parts) >= 9:
                pts = []
                for i in range(1, 9, 2):
                    pts.append({"x": float(parts[i]), "y": float(parts[i + 1])})
                label["type"] = "obb"
                label["points"] = pts
            elif len(parts) >= 3:
                label["type"] = "centroid"
                label["x"] = float(parts[1])
                label["y"] = float(parts[2])
                label["w"] = float(parts[3]) if len(parts) > 3 else 0.0
                label["h"] = float(parts[4]) if len(parts) > 4 else 0.0
            else:
                label["type"] = "unknown"
                label["raw"] = raw.strip()
            labels.append(label)
    return labels


class Handler(BaseHTTPRequestHandler):
    state: AppState

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        try:
            self.route_get()
        except FileNotFoundError:
            self.send_error(404, "Not found")
        except Exception as exc:
            self.send_error(500, str(exc))

    def route_get(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        qs = parse_qs(parsed.query)

        if path == "/":
            self.send_file(STATIC_DIR / "index.html")
        elif path.startswith("/static/"):
            self.send_file(STATIC_DIR / path.removeprefix("/static/"))
        elif path == "/api/models":
            models = [model_summary(self.state, model_dir) for model_dir in self.state.model_dirs()]
            self.send_json({"models": models, "classes": self.state.classes})
        elif path == "/api/model":
            model_id = require_one(qs, "id")
            model_dir = self.state.model_dir(model_id)
            inspection = read_inspection(model_dir)
            if inspection is not None:
                records = inspection.get("records", [])
                self.send_json({
                    "model": model_summary(self.state, model_dir),
                    "thresholds": inspection.get("thresholds", {}),
                    "rows": [
                        normalize_inspection_record(record, i)
                        for i, record in enumerate(records)
                    ],
                })
            else:
                rows = read_report(model_dir / "error_inspection/error_report.csv")
                self.send_json({
                    "model": model_summary(self.state, model_dir),
                    "rows": [normalize_row(row, i) for i, row in enumerate(rows)],
                })
        elif path == "/api/labels":
            image_name = require_one(qs, "image")
            label_kind = qs.get("kind", ["obb"])[0]
            labels_dir = self.state.centroid_labels_dir if label_kind == "centroid" else self.state.labels_dir
            label_path = labels_dir / (Path(image_name).stem + ".txt")
            ensure_under(label_path.resolve(), labels_dir)
            self.send_json({
                "image": image_name,
                "kind": label_kind,
                "labels": parse_label_file(label_path, self.state.classes),
                "path": str(label_path),
            })
        elif path == "/asset/raw":
            image_name = require_one(qs, "image")
            image_path = (self.state.images_dir / Path(image_name).name).resolve()
            ensure_under(image_path, self.state.images_dir)
            self.send_file(image_path)
        elif path == "/asset/annotated":
            model_id = require_one(qs, "model")
            image_name = require_one(qs, "image")
            model_dir = self.state.model_dir(model_id)
            requested = Path(image_name)
            if len(requested.parts) > 1:
                image_path = (model_dir / requested).resolve()
                ensure_under(image_path, model_dir)
            else:
                json_path = (model_dir / "annotated_images" / requested.name).resolve()
                legacy_path = (model_dir / "error_inspection" / requested.name).resolve()
                image_path = json_path if json_path.is_file() else legacy_path
                ensure_under(image_path, model_dir)
            self.send_file(image_path)
        elif path == "/asset/inspection":
            model_id = require_one(qs, "model")
            asset_path = require_one(qs, "path")
            model_dir = self.state.model_dir(model_id)
            image_path = (model_dir / asset_path).resolve()
            ensure_under(image_path, model_dir)
            self.send_file(image_path)
        else:
            self.send_error(404, "Not found")

    def send_json(self, payload: object) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path) -> None:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def require_one(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not values or values[0] == "":
        raise ValueError(f"Missing required query parameter: {name}")
    return values[0]


def main() -> None:
    args = parse_args()
    Handler.state = AppState(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Serving error inspection app at {url}")
    print(f"Scanning results under {Handler.state.results_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
