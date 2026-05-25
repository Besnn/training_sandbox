# YOLO-OBB Annotation Viewer

Run from `traffic_signal_detection/src`:

```bash
python3 annotation_viewer/app.py
```

Open:

```text
http://127.0.0.1:8765
```

By default it shows:

- `datasets/split_obb_dataset/val/images`
- `datasets/split_obb_dataset/val/labels`

Use another split:

```bash
python3 annotation_viewer/app.py \
  --images datasets/split_obb_dataset/train/images \
  --labels datasets/split_obb_dataset/train/labels
```

The viewer supports searching, filtering annotated/empty-label images, filtering
by class, and keyboard navigation with left/right arrows.
