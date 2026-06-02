from ultralytics import YOLO
import cv2

img = cv2.imread("datasets/yolo_pl_test/images/snap_1026_20260527_102101_834.jpg")
model = YOLO("models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx")

results = model(img[:-192, 256:],
    conf=0.5,
    iou=0.45,
)[0]

obb = results.obb
print(f"{len(obb)} detections after NMS (conf≥0.25, iou≤0.45)")
for i, (cls, conf) in enumerate(zip(obb.cls, obb.conf)):
    print(f"  [{i}] {model.names[int(cls)]}  conf={conf:.3f}")

results.show()
