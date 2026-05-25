#!/usr/bin/env python3
"""Local YOLO-OBB annotation viewer.

Run from traffic_signal_detection/src:
    python3 annotation_viewer/app.py

Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "datasets/split_obb_dataset/train/images"
DEFAULT_LABELS = ROOT / "datasets/split_obb_dataset/train/labels"
DEFAULT_CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>YOLO-OBB Annotation Viewer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #68758a;
      --line: #d7dee9;
      --blue: #245a86;
      --blue-soft: #e8f1f9;
      --red: #d33f49;
      --green: #138a63;
      --yellow: #b98100;
      --purple: #7651c9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
      height: 100vh;
      overflow: hidden;
    }
    .app {
      display: grid;
      grid-template-columns: 340px 1fr;
      height: 100vh;
      min-width: 900px;
    }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--line);
      display: grid;
      grid-template-rows: auto auto 1fr;
      min-height: 0;
    }
    header {
      padding: 18px 18px 14px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      font-size: 18px;
      line-height: 1.2;
      margin: 0 0 8px;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      word-break: break-word;
    }
    .controls {
      display: grid;
      gap: 10px;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
    }
    input[type="search"], select {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      height: 36px;
      border-radius: 6px;
      padding: 0 10px;
      font-size: 14px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .checks {
      display: flex;
      gap: 12px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }
    label { user-select: none; }
    .list {
      overflow: auto;
      min-height: 0;
    }
    .item {
      border: 0;
      border-bottom: 1px solid var(--line);
      background: transparent;
      width: 100%;
      text-align: left;
      padding: 10px 14px 10px 18px;
      cursor: pointer;
      color: var(--ink);
      display: grid;
      gap: 4px;
    }
    .item:hover, .item.active { background: var(--blue-soft); }
    .name {
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }
    main {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .toolbar {
      height: 58px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 18px;
    }
    .title {
      min-width: 0;
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .buttons {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-shrink: 0;
    }
    button.nav {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      height: 34px;
      min-width: 38px;
      padding: 0 10px;
      font-size: 14px;
      cursor: pointer;
    }
    button.nav:hover { border-color: var(--blue); }
    .stage {
      position: relative;
      min-width: 0;
      min-height: 0;
      overflow: auto;
      display: grid;
      place-items: center;
      padding: 18px;
      background: #e9edf4;
    }
    .canvas-wrap {
      position: relative;
      max-width: 100%;
      max-height: 100%;
      box-shadow: 0 10px 26px rgba(23, 32, 51, 0.18);
      background: #111;
    }
    canvas { display: block; max-width: 100%; max-height: calc(100vh - 160px); }
    .details {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 10px 18px;
      min-height: 82px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
    }
    .chips {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 13px;
      background: #fff;
    }
    .raw {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
      max-height: 60px;
      overflow: auto;
      white-space: pre-wrap;
    }
    .empty {
      color: var(--muted);
      font-size: 14px;
    }
    @media (max-width: 920px) {
      body { overflow: auto; }
      .app { grid-template-columns: 1fr; min-width: 0; height: auto; min-height: 100vh; }
      aside { max-height: 42vh; }
      canvas { max-height: 55vh; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>YOLO-OBB Annotation Viewer</h1>
        <div id="paths" class="sub">Loading dataset...</div>
      </header>
      <section class="controls">
        <input id="search" type="search" placeholder="Search image name" />
        <div class="row">
          <select id="labelFilter">
            <option value="all">All images</option>
            <option value="annotated">Annotated only</option>
            <option value="empty">Empty labels only</option>
          </select>
          <select id="classFilter">
            <option value="all">All classes</option>
          </select>
        </div>
        <div class="checks">
          <label><input id="showLabels" type="checkbox" checked /> labels</label>
          <label><input id="showPoints" type="checkbox" checked /> vertices</label>
          <label><input id="dimImage" type="checkbox" /> dim image</label>
        </div>
      </section>
      <section id="list" class="list"></section>
    </aside>
    <main>
      <section class="toolbar">
        <div id="title" class="title">No image selected</div>
        <div class="buttons">
          <button class="nav" id="prev" title="Previous image">Prev</button>
          <button class="nav" id="next" title="Next image">Next</button>
          <button class="nav" id="fit" title="Fit image">Fit</button>
        </div>
      </section>
      <section class="stage">
        <div class="canvas-wrap">
          <canvas id="canvas"></canvas>
        </div>
      </section>
      <section class="details">
        <div>
          <div id="chips" class="chips"></div>
          <div id="raw" class="raw"></div>
        </div>
        <div id="count" class="empty"></div>
      </section>
    </main>
  </div>
  <script>
    const colors = ["#d33f49", "#138a63", "#245a86", "#b98100", "#7651c9", "#007c89"];
    const state = {
      images: [],
      filtered: [],
      selected: -1,
      annotations: [],
      classes: [],
      image: new Image(),
      scale: 1,
    };

    const el = {
      paths: document.getElementById("paths"),
      search: document.getElementById("search"),
      labelFilter: document.getElementById("labelFilter"),
      classFilter: document.getElementById("classFilter"),
      showLabels: document.getElementById("showLabels"),
      showPoints: document.getElementById("showPoints"),
      dimImage: document.getElementById("dimImage"),
      list: document.getElementById("list"),
      title: document.getElementById("title"),
      prev: document.getElementById("prev"),
      next: document.getElementById("next"),
      fit: document.getElementById("fit"),
      canvas: document.getElementById("canvas"),
      chips: document.getElementById("chips"),
      raw: document.getElementById("raw"),
      count: document.getElementById("count"),
    };
    const ctx = el.canvas.getContext("2d");

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function className(id) {
      return state.classes[id] ?? `class-${id}`;
    }

    function imageUrl(file) {
      return `/image?file=${encodeURIComponent(file)}`;
    }

    function applyFilters() {
      const query = el.search.value.trim().toLowerCase();
      const labelFilter = el.labelFilter.value;
      const classFilter = el.classFilter.value;
      state.filtered = state.images.filter((img) => {
        if (query && !img.file.toLowerCase().includes(query)) return false;
        if (labelFilter === "annotated" && img.annotation_count === 0) return false;
        if (labelFilter === "empty" && img.annotation_count !== 0) return false;
        if (classFilter !== "all" && !img.classes.includes(Number(classFilter))) return false;
        return true;
      });
      renderList();
      if (!state.filtered.length) {
        state.selected = -1;
        drawEmpty();
        return;
      }
      if (state.selected < 0 || state.selected >= state.filtered.length) {
        selectIndex(0);
      }
    }

    function renderList() {
      el.list.innerHTML = "";
      state.filtered.forEach((img, index) => {
        const button = document.createElement("button");
        button.className = `item${index === state.selected ? " active" : ""}`;
        button.type = "button";
        button.onclick = () => selectIndex(index);
        const classes = img.classes.map(className).join(", ") || "no boxes";
        button.innerHTML = `
          <div class="name">${img.file}</div>
          <div class="meta"><span>${img.annotation_count} box${img.annotation_count === 1 ? "" : "es"}</span><span>${classes}</span></div>
        `;
        el.list.appendChild(button);
      });
      el.count.textContent = `${state.filtered.length} / ${state.images.length} images`;
    }

    async function selectIndex(index) {
      if (index < 0 || index >= state.filtered.length) return;
      state.selected = index;
      renderList();
      const item = state.filtered[index];
      el.title.textContent = item.file;
      const data = await api(`/api/annotations?file=${encodeURIComponent(item.file)}`);
      state.annotations = data.annotations;
      state.image = new Image();
      state.image.onload = () => {
        fitCanvas();
        renderDetails(data.raw_lines);
      };
      state.image.src = imageUrl(item.file);
    }

    function fitCanvas() {
      if (!state.image.naturalWidth) return;
      const maxW = Math.max(320, window.innerWidth - 400);
      const maxH = Math.max(240, window.innerHeight - 160);
      state.scale = Math.min(1, maxW / state.image.naturalWidth, maxH / state.image.naturalHeight);
      el.canvas.width = Math.round(state.image.naturalWidth * state.scale);
      el.canvas.height = Math.round(state.image.naturalHeight * state.scale);
      draw();
    }

    function drawEmpty() {
      el.title.textContent = "No image selected";
      el.canvas.width = 900;
      el.canvas.height = 540;
      ctx.fillStyle = "#e9edf4";
      ctx.fillRect(0, 0, el.canvas.width, el.canvas.height);
      ctx.fillStyle = "#68758a";
      ctx.font = "18px system-ui";
      ctx.textAlign = "center";
      ctx.fillText("No images match the current filters", el.canvas.width / 2, el.canvas.height / 2);
      el.chips.innerHTML = "";
      el.raw.textContent = "";
    }

    function draw() {
      if (!state.image.naturalWidth) return;
      const s = state.scale;
      ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
      ctx.drawImage(state.image, 0, 0, el.canvas.width, el.canvas.height);
      if (el.dimImage.checked) {
        ctx.fillStyle = "rgba(0, 0, 0, 0.32)";
        ctx.fillRect(0, 0, el.canvas.width, el.canvas.height);
      }
      state.annotations.forEach((ann, idx) => {
        const color = colors[ann.class_id % colors.length];
        const pts = ann.points.map(([x, y]) => [x * el.canvas.width, y * el.canvas.height]);
        ctx.beginPath();
        pts.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
        ctx.closePath();
        ctx.fillStyle = `${color}33`;
        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, 3 * s);
        ctx.fill();
        ctx.stroke();
        if (el.showPoints.checked) {
          ctx.fillStyle = color;
          pts.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x, y, Math.max(3, 4 * s), 0, Math.PI * 2);
            ctx.fill();
          });
        }
        if (el.showLabels.checked) {
          const minX = Math.min(...pts.map(p => p[0]));
          const minY = Math.min(...pts.map(p => p[1]));
          const text = `${className(ann.class_id)} #${idx + 1}`;
          ctx.font = "600 14px system-ui";
          const width = ctx.measureText(text).width + 12;
          const y = Math.max(22, minY - 8);
          ctx.fillStyle = color;
          ctx.fillRect(minX, y - 19, width, 22);
          ctx.fillStyle = "white";
          ctx.fillText(text, minX + 6, y - 4);
        }
      });
    }

    function renderDetails(rawLines) {
      const counts = new Map();
      state.annotations.forEach((ann) => counts.set(ann.class_id, (counts.get(ann.class_id) || 0) + 1));
      el.chips.innerHTML = "";
      if (!state.annotations.length) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = "No annotations";
        el.chips.appendChild(chip);
      } else {
        [...counts.entries()].sort((a, b) => a[0] - b[0]).forEach(([id, count]) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = `${className(id)}: ${count}`;
          el.chips.appendChild(chip);
        });
      }
      el.raw.textContent = rawLines.join("\n");
      draw();
    }

    el.search.oninput = applyFilters;
    el.labelFilter.onchange = applyFilters;
    el.classFilter.onchange = applyFilters;
    el.showLabels.onchange = draw;
    el.showPoints.onchange = draw;
    el.dimImage.onchange = draw;
    el.prev.onclick = () => selectIndex(Math.max(0, state.selected - 1));
    el.next.onclick = () => selectIndex(Math.min(state.filtered.length - 1, state.selected + 1));
    el.fit.onclick = fitCanvas;
    window.onresize = fitCanvas;
    window.onkeydown = (event) => {
      if (event.key === "ArrowLeft") selectIndex(Math.max(0, state.selected - 1));
      if (event.key === "ArrowRight") selectIndex(Math.min(state.filtered.length - 1, state.selected + 1));
    };

    async function init() {
      const data = await api("/api/images");
      state.images = data.images;
      state.classes = data.classes;
      el.paths.textContent = `${data.images_dir} | ${data.labels_dir}`;
      state.classes.forEach((name, id) => {
        const option = document.createElement("option");
        option.value = String(id);
        option.textContent = name;
        el.classFilter.appendChild(option);
      });
      applyFilters();
    }

    init().catch((err) => {
      el.paths.textContent = err.message;
      drawEmpty();
    });
  </script>
</body>
</html>
"""


def parse_label_file(path: Path) -> tuple[list[dict], list[str]]:
    annotations = []
    raw_lines = []
    if not path.is_file():
        return annotations, raw_lines

    for line_no, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        raw_lines.append(line)
        parts = line.split()
        if len(parts) < 9:
            annotations.append({"error": f"line {line_no}: expected 9 values", "raw": line})
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(v) for v in parts[1:9]]
        except ValueError:
            annotations.append({"error": f"line {line_no}: non-numeric value", "raw": line})
            continue
        points = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        annotations.append({"class_id": class_id, "points": points, "raw": line})
    return annotations, raw_lines


def safe_file(images_dir: Path, rel_file: str) -> Path:
    rel_path = Path(unquote(rel_file))
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError("invalid image path")
    image_path = (images_dir / rel_path).resolve()
    if images_dir.resolve() not in image_path.parents and image_path != images_dir.resolve():
        raise ValueError("image path escapes image directory")
    return image_path


class AnnotationServer(BaseHTTPRequestHandler):
    images_dir: Path
    labels_dir: Path
    classes: list[str]

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload).encode("utf-8"), "application/json", status)

    def send_text(self, text: str, status: int = 200) -> None:
        self.send_bytes(text.encode("utf-8"), "text/plain; charset=utf-8", status)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/api/images":
                self.handle_images()
            elif parsed.path == "/api/annotations":
                self.handle_annotations(parsed.query)
            elif parsed.path == "/image":
                self.handle_image(parsed.query)
            else:
                self.send_text("not found", 404)
        except Exception as exc:  # noqa: BLE001
            self.send_json({"error": str(exc)}, 500)

    def handle_images(self) -> None:
        images = []
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            annotations, _ = parse_label_file(label_path)
            class_ids = sorted({
                ann["class_id"] for ann in annotations if "class_id" in ann
            })
            images.append({
                "file": image_path.name,
                "annotation_count": len([ann for ann in annotations if "class_id" in ann]),
                "classes": class_ids,
            })
        self.send_json({
            "images_dir": str(self.images_dir),
            "labels_dir": str(self.labels_dir),
            "classes": self.classes,
            "images": images,
        })

    def handle_annotations(self, query: str) -> None:
        file_name = parse_qs(query).get("file", [""])[0]
        image_path = safe_file(self.images_dir, file_name)
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        annotations, raw_lines = parse_label_file(label_path)
        self.send_json({
            "file": image_path.name,
            "label_file": str(label_path),
            "annotations": annotations,
            "raw_lines": raw_lines,
        })

    def handle_image(self, query: str) -> None:
        file_name = parse_qs(query).get("file", [""])[0]
        image_path = safe_file(self.images_dir, file_name)
        if not image_path.is_file():
            self.send_text("image not found", 404)
            return
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        self.send_bytes(image_path.read_bytes(), content_type)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=str(DEFAULT_IMAGES), help="Directory of images.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Directory of YOLO-OBB labels.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="Class names in label order.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8765, help="Server port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images).expanduser().resolve()
    labels_dir = Path(args.labels).expanduser().resolve()
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"Labels directory not found: {labels_dir}")

    handler = type(
        "ConfiguredAnnotationServer",
        (AnnotationServer,),
        {"images_dir": images_dir, "labels_dir": labels_dir, "classes": args.classes},
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving annotation viewer at http://{args.host}:{args.port}")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
