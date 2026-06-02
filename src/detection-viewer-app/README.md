# Detection Viewer App

Detection-only UI for `inspection.json` folders.

```bash
cd examples/benchmarks/detection-viewer-app
python3 server.py
```

Open:

```text
http://127.0.0.1:8766
```

The app scans `../benchmark_results/inspection_folders` by default. It ignores
ground-truth rendering and shows model detections only, with toggles for class,
status, confidence, and label text.
