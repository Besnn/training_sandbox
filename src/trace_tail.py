"""Find every node downstream of the cv3.*.2 class-head Convs and print
their values in FP32 vs INT8 — pinpoint where the class signal dies."""
import numpy as np
import onnx
import onnxruntime as ort
import cv2

IMG = "datasets/yolo_pl_test/images/snap_1026_20260527_102101_834.jpg"


def preprocess(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    return np.expand_dims(img, 0).astype(np.float32) / 255.0


# Load the INT8 graph and walk downstream from cv3.X.2 outputs.
m = onnx.load("models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx")
seeds = [
    "/model.22/cv3.0/cv3.0.2/Conv_output_0",
    "/model.22/cv3.1/cv3.1.2/Conv_output_0",
    "/model.22/cv3.2/cv3.2.2/Conv_output_0",
]
producer = {}
for node in m.graph.node:
    for out in node.output:
        producer[out] = node

# BFS forward: find all tensors reachable from seeds.
consumers_by_tensor = {}
for node in m.graph.node:
    for inp in node.input:
        consumers_by_tensor.setdefault(inp, []).append(node)

reachable = set(seeds)
frontier = list(seeds)
ordered_tensors = list(seeds)
while frontier:
    t = frontier.pop(0)
    for n in consumers_by_tensor.get(t, []):
        for out in n.output:
            if out and out not in reachable:
                reachable.add(out)
                frontier.append(out)
                ordered_tensors.append(out)

print(f"{len(ordered_tensors)} tensors downstream of class-head final convs")

# Load both traced models (already created by trace_int8.py).
s_fp = ort.InferenceSession("/tmp/fp32_traced.onnx", providers=["CPUExecutionProvider"])
s_i8 = ort.InferenceSession("/tmp/int8_traced.onnx", providers=["CPUExecutionProvider"])

x = preprocess(IMG)
fp = dict(zip([o.name for o in s_fp.get_outputs()],
              s_fp.run(None, {s_fp.get_inputs()[0].name: x})))
i8 = dict(zip([o.name for o in s_i8.get_outputs()],
              s_i8.run(None, {s_i8.get_inputs()[0].name: x})))

print(f"\n{'name':<70} {'op':<15} {'shape':<22} {'fp_max':>10} {'i8_max':>10}")
print("-" * 130)

for t in ordered_tensors:
    if t in fp and t in i8 and fp[t].shape == i8[t].shape:
        op = producer.get(t).op_type if t in producer else "graph_input"
        fp_v = fp[t]
        i8_v = i8[t]
        print(f"{t[:70]:<70} {op:<15} {str(fp_v.shape):<22} {fp_v.max():>10.4f} {i8_v.max():>10.4f}")
