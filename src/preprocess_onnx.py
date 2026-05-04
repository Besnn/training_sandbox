from onnxruntime.quantization import quant_pre_process

MODEL_FP32 = "models/best.onnx"
MODEL_PREPROCESSED = "models/best_preprocessed.onnx"

quant_pre_process(
    input_model_path=MODEL_FP32,
    output_model_path=MODEL_PREPROCESSED,
    skip_optimization=False,
    skip_onnx_shape=False,
    skip_symbolic_shape=False,
)