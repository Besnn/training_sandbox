import onnx

model = onnx.load("models/best_int8.onnx")
ops = {}

for node in model.graph.node:
    ops[node.op_type] = ops.get(node.op_type, 0) + 1

for k, v in sorted(ops.items()):
    if "Quant" in k or "Dequant" in k or "QLinear" in k or "Integer" in k:
        print(k, v)