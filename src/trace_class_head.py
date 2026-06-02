"""Look specifically at the class-head's final Convs and the Sigmoid/Concat
that follow, comparing FP32 vs INT8."""
import numpy as np
import cv2
import onnxruntime as ort

IMG = "datasets/yolo_pl_test/images/snap_1026_20260527_102101_834.jpg"


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return np.expand_dims(img, 0).astype(np.float32) / 255.0


x = preprocess(IMG)

s_fp = ort.InferenceSession("/tmp/fp32_traced.onnx", providers=["CPUExecutionProvider"])
s_i8 = ort.InferenceSession("/tmp/int8_traced.onnx", providers=["CPUExecutionProvider"])

fp_names = [o.name for o in s_fp.get_outputs()]
i8_names = [o.name for o in s_i8.get_outputs()]

fp_out = s_fp.run(None, {s_fp.get_inputs()[0].name: x})
i8_out = s_i8.run(None, {s_i8.get_inputs()[0].name: x})

fp_map = dict(zip(fp_names, fp_out))
i8_map = dict(zip(i8_names, i8_out))

# Find anything related to cv3 (class head in YOLOv8 OBB) and the final concat
keywords = ["cv3", "Sigmoid", "Concat_5", "Concat_4", "Concat_3", "Reshape_2", "Reshape_3", "output0"]

print(f"{'name':<70} {'shape':<25} {'fp_max':>10} {'i8_max':>10} {'fp_mean':>10} {'i8_mean':>10}")
print("-" * 145)
for name in sorted(fp_map.keys()):
    if any(k in name for k in keywords):
        if name in i8_map and fp_map[name].shape == i8_map[name].shape:
            fp = fp_map[name]
            i8 = i8_map[name]
            print(f"{name[:70]:<70} {str(fp.shape):<25} {fp.max():>10.4f} {i8.max():>10.4f} {fp.mean():>10.4f} {i8.mean():>10.4f}")

# Also dump the actual final output tensor name
print("\n=== Final graph outputs ===")
for name in fp_names:
    if "output" in name.lower() and name in i8_map:
        fp = fp_map[name]
        i8 = i8_map[name]
        if fp.shape == i8.shape:
            print(f"{name}: shape={fp.shape} fp_max={fp.max():.4f} i8_max={i8.max():.4f}")
