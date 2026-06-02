import numpy as np
import cv2
import onnxruntime as ort

IMG = "datasets/yolo_pl_test/images/snap_1026_20260527_102101_834.jpg"
FP32 = "models/yolov8n-obb-onnx/yolov8n-obb-fp32.onnx"
INT8 = "models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx"
IMG_SIZE = 640


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0).astype(np.float32) / 255.0


def run(path, x):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    out = sess.run(None, {name: x})[0]
    return out


x = preprocess(IMG)
print(f"input: shape={x.shape} dtype={x.dtype} range=[{x.min():.3f}, {x.max():.3f}]")

for tag, path in [("FP32", FP32), ("INT8", INT8)]:
    y = run(path, x)
    print(f"\n[{tag}] output shape: {y.shape}, dtype: {y.dtype}")
    # YOLOv8-OBB output layout: (1, 6+nc, N) -> [cx, cy, w, h, *cls_scores, angle]
    # For 15-class OBB (DOTA), that's (1, 20, 8400). For your custom model, nc differs.
    # The class scores live in channels 4 .. -1 (last channel is angle).
    cls = y[0, 4:-1, :]  # (nc, N)
    angle = y[0, -1, :]  # (N,)
    max_conf_per_box = cls.max(axis=0)  # (N,)
    print(f"  num predictions: {y.shape[-1]}")
    print(f"  class-score range: [{cls.min():.4f}, {cls.max():.4f}]")
    print(f"  angle range: [{angle.min():.4f}, {angle.max():.4f}]")
    print(f"  top-10 box confidences: {np.sort(max_conf_per_box)[-10:][::-1]}")
    print(f"  boxes above conf 0.05: {(max_conf_per_box > 0.05).sum()}")
    print(f"  boxes above conf 0.01: {(max_conf_per_box > 0.01).sum()}")
    print(f"  boxes above conf 0.001: {(max_conf_per_box > 0.001).sum()}")
