from onnxruntime.quantization import quant_pre_process

MODEL_FP32 = "models/yolov8n-obb-onnx/yolov8n-obb-fp32.onnx"
MODEL_PREPROCESSED = "models/yolov8n-obb-onnx-quant-preprocessed.onnx"

quant_pre_process(
    input_model_path=MODEL_FP32,
    output_model_path=MODEL_PREPROCESSED,
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=False,
)