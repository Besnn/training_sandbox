# Error Inspection App

Local web UI for browsing `benchmark_results/inspection_folders/*/inspection.json`
artifacts. Legacy `error_inspection/error_report.csv` folders are still
supported.

```bash
cd examples/benchmarks/error-inspection-app
python3 server.py
```

Open:

```text
http://127.0.0.1:8765
```

The app scans `../benchmark_results/inspection_folders` for `inspection.json`,
serves images from each inspection folder, and overlays ground-truth labels from
`../datasets/yolo_pl_test/labels` on the original validation frames.

Useful options:

```bash
python3 server.py \
  --results-root ../benchmark_results/inspection_folders \
  --images ../datasets/yolo_pl_test/images \
  --labels ../datasets/yolo_pl_test/labels \
  --centroid-labels ../datasets/yolo_pl_test_centroid/labels \
  --port 8765
```
