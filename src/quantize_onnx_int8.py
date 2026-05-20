import cv2
import numpy as np
import onnx
from pathlib import Path
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
    quantize_static,
    quant_pre_process
)

MODEL_FP32 = "models/yolov8n-obb-onnx-quant-preprocessed.onnx"
MODEL_INT8 = "models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx"
CALIB_DIR = "datasets/split_obb_dataset/train/images"

IMG_SIZE = 640
NUM_CALIB_IMAGES = 100


class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, img_size=640, max_images=100):
        self.image_paths = []

        image_dir = Path(image_dir)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.image_paths.extend(image_dir.glob(ext))

        self.image_paths = self.image_paths[:max_images]
        self.input_name = input_name
        self.img_size = img_size
        self.index = 0

        if not self.image_paths:
            raise RuntimeError(f"No calibration images found in {image_dir}")

    def preprocess(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        return img

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        return {
            self.input_name: self.preprocess(image_path)
        }


def get_input_name(model_path):
    model = onnx.load(model_path)
    return model.graph.input[0].name


if __name__ == "__main__":
    input_name = get_input_name(MODEL_FP32)

    print(f"Input name: {input_name}")
    print(f"Quantizing: {MODEL_FP32} -> {MODEL_INT8}")

    calibration_reader = YOLOCalibrationDataReader(
        image_dir=CALIB_DIR,
        input_name=input_name,
        img_size=IMG_SIZE,
        max_images=NUM_CALIB_IMAGES,
    )

    quantize_static(
        model_input=MODEL_FP32,
        model_output=MODEL_INT8,
        calibration_data_reader=calibration_reader,

        # QDQ is usually the safest modern format.
        quant_format=QuantFormat.QDQ,

        # Common default for CPU/runtime compatibility.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,

        # MinMax is simple and stable. Entropy can also be tested.
        calibrate_method=CalibrationMethod.MinMax,

        # Per-channel usually helps Conv accuracy.
        per_channel=True,

        # Keep external data disabled unless your model is huge.
        use_external_data_format=False,
    )

    print("Done.")