#!/usr/bin/env python3
"""Serve a detection-only viewer for inspection.json folders."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


APP_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_RESULTS_ROOT = BENCHMARKS_DIR / "benchmark_results/detection_folders"
DEFAULT_IMAGES_DIR = BENCHMARKS_DIR / "datasets/yolo_pl_test/images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                        help="Directory containing */inspection.json folders.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES_DIR,
                        help="Fallback raw image directory.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    return parser.parse_args()


class AppState:
    def __init__(self, args: argparse.Namespace):
        self.results_root = args.results_root.resolve()
        self.images_dir = args.images.resolve()

    def model_dirs(self) -> list[Path]:
        return sorted(path.parent for path in self.results_root.rglob("inspection.json"))

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


def read_inspection(model_dir: Path) -> dict:
    path = model_dir / "inspection.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def model_summary(state: AppState, model_dir: Path) -> dict:
    inspection = read_inspection(model_dir)
    model = inspection.get("model", {})
    records = inspection.get("records", [])
    detection_count = sum(len(record.get("detections", []) or []) for record in records)
    return {
        "id": state.model_id(model_dir),
        "name": model.get("name") or model_dir.name,
        "kind": model.get("kind", ""),
        "images": len(records),
        "detections": detection_count,
        "thresholds": inspection.get("thresholds", {}),
    }


def normalize_record(record: dict, index: int) -> dict:
    detections = record.get("detections", []) or []
    return {
        "index": index,
        "image": record.get("image", ""),
        "rawImage": record.get("rawImage", ""),
        "width": record.get("width", 0),
        "height": record.get("height", 0),
        "detections": detections,
        "detectionCount": len(detections),
        "classes": sorted({det.get("className", str(det.get("classId", ""))) for det in detections}),
    }


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
            self.send_json({"models": models})
        elif path == "/api/model":
            model_id = require_one(qs, "id")
            model_dir = self.state.model_dir(model_id)
            inspection = read_inspection(model_dir)
            self.send_json({
                "model": model_summary(self.state, model_dir),
                "classes": inspection.get("classes", []),
                "thresholds": inspection.get("thresholds", {}),
                "records": [
                    normalize_record(record, index)
                    for index, record in enumerate(inspection.get("records", []))
                ],
            })
        elif path == "/asset/raw":
            model_id = require_one(qs, "model")
            raw_path = require_one(qs, "path")
            image_name = require_one(qs, "image")
            model_dir = self.state.model_dir(model_id)
            if raw_path and not raw_path.startswith("/"):
                image_path = (model_dir / raw_path).resolve()
                ensure_under(image_path, model_dir)
            else:
                image_path = (self.state.images_dir / Path(image_name).name).resolve()
                ensure_under(image_path, self.state.images_dir)
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
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
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
    print(f"Serving detection viewer at http://{args.host}:{args.port}")
    print(f"Scanning {Handler.state.results_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down")


if __name__ == "__main__":
    main()
