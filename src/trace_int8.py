"""Walk through the INT8 model's intermediate tensors and find where
the class-head signal dies relative to FP32."""
import numpy as np
import cv2
import onnx
import onnxruntime as ort

IMG = "datasets/yolo_pl_test/images/snap_1026_20260527_102101_834.jpg"
FP32 = "models/tmp/yolov8n-onnx-quant-preprocessed.onnx"
INT8 = "models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx"


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return np.expand_dims(img, 0).astype(np.float32) / 255.0


def expose_all_intermediates(model_path, out_path):
    m = onnx.load(model_path)
    existing_outputs = {o.name for o in m.graph.output}
    # Collect every node output that isn't already a graph output.
    extra = []
    for node in m.graph.node:
        # Skip ops that produce non-float tensors when we want a clean comparison.
        if node.op_type in ("QuantizeLinear",):
            continue
        for out in node.output:
            if out and out not in existing_outputs:
                extra.append(onnx.helper.make_tensor_value_info(out, onnx.TensorProto.FLOAT, None))
    m.graph.output.extend(extra)
    onnx.save(m, out_path)
    return [o.name for o in m.graph.output]


print("Exposing intermediates...")
fp_outs = expose_all_intermediates(FP32, "/tmp/fp32_traced.onnx")
i8_outs = expose_all_intermediates(INT8, "/tmp/int8_traced.onnx")

x = preprocess(IMG)

print("Running FP32...")
s_fp = ort.InferenceSession("/tmp/fp32_traced.onnx", providers=["CPUExecutionProvider"])
fp_results = s_fp.run(None, {s_fp.get_inputs()[0].name: x})
fp_map = dict(zip([o.name for o in s_fp.get_outputs()], fp_results))

print("Running INT8...")
s_i8 = ort.InferenceSession("/tmp/int8_traced.onnx", providers=["CPUExecutionProvider"])
i8_results = s_i8.run(None, {s_i8.get_inputs()[0].name: x})
i8_map = dict(zip([o.name for o in s_i8.get_outputs()], i8_results))

# Find tensors that exist in both with matching shapes.
common = []
for name in fp_map:
    if name in i8_map and fp_map[name].shape == i8_map[name].shape:
        common.append(name)

print(f"\n{len(common)} tensors with matching shape\n")

# Compute relative magnitude ratio per tensor (int8_max / fp32_max).
# Tensors where this ratio collapses to ~0 are where the signal dies.
rows = []
for name in common:
    fp_abs = np.abs(fp_map[name]).max()
    i8_abs = np.abs(i8_map[name]).max()
    if fp_abs > 1e-6:
        ratio = i8_abs / fp_abs
        rows.append((name, fp_map[name].shape, fp_abs, i8_abs, ratio))

# Sort by ratio ascending — the worst (most collapsed) at the top.
rows.sort(key=lambda r: r[4])

print("=== 30 tensors where INT8 signal collapsed most vs FP32 ===")
print(f"{'name':<60} {'shape':<25} {'fp_max':>10} {'i8_max':>10} {'ratio':>8}")
for r in rows[:30]:
    name, shape, fp, i8, ratio = r
    print(f"{name[:60]:<60} {str(shape):<25} {fp:>10.4f} {i8:>10.4f} {ratio:>8.4f}")
