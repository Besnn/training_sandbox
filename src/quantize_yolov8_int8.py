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

MODEL_FP32 = "models/tmp/yolov8n-onnx-quant-preprocessed.onnx"
MODEL_INT8 = "models/yolov8n-obb-onnx/yolov8n-obb-int8.onnx"
CALIB_DIR = "datasets/split_obb_dataset/train/images"

IMG_SIZE = 640
NUM_CALIB_IMAGES = 150

# Keep the ENTIRE OBB detect head (module model.22) in FP32. Quantizing it
# does two kinds of damage:
#   - the class branch shares the final Concat's uint8 scale with box coords
#     (~660) and angle (~pi), which crushes class scores to zero; and
#   - the box-regression (cv2 / DFL) and angle (cv4) branches lose coordinate
#     precision, which tanks IoU for long, thin, rotated boxes (railroad-
#     crossing, trefolo) while compact classes (lights-on/off) shrug it off.
# The backbone + neck (where almost all the size/latency win lives) stay int8.
# NOTE: it is model.22 in YOLOv8 (model.23 in YOLO26), and quantize_static
# silently ignores unmatched names — the guard below catches a wrong prefix.
HEAD_PREFIX = "/model.22/"


def head_nodes(model_path, prefix=HEAD_PREFIX):
    graph = onnx.load(model_path).graph
    return [n.name for n in graph.node if n.name.startswith(prefix)]


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


def check_exclude_nodes(model_path, nodes_to_exclude):
    """Fail loudly if the exclude list is empty or names don't match the graph.

    quantize_static silently ignores unmatched names; an empty/typo'd list
    quantizes the OBB head and collapses detections or localization.
    """
    if not nodes_to_exclude:
        raise SystemExit(
            f"nodes_to_exclude is empty — no node matched prefix {HEAD_PREFIX!r}. "
            "The head would be quantized and detections would collapse."
        )
    graph_nodes = {n.name for n in onnx.load(model_path).graph.node}
    missing = [n for n in nodes_to_exclude if n not in graph_nodes]
    if missing:
        raise SystemExit(
            "nodes_to_exclude not found in graph (would be silently ignored): "
            f"{missing}"
        )


if __name__ == "__main__":
    input_name = get_input_name(MODEL_FP32)
    nodes_to_exclude = head_nodes(MODEL_FP32)
    check_exclude_nodes(MODEL_FP32, nodes_to_exclude)

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

        # Percentile is far more robust than MinMax for detection heads —
        # MinMax let a single calibration outlier crush all class-score
        # signal to zero. Entropy is the other reasonable choice.
        calibrate_method=CalibrationMethod.Entropy,
        # extra_options={"CalibPercentile": 99.999},
        extra_options={"NumBins": 512},

        # Per-channel usually helps Conv accuracy.
        per_channel=True,

        # Keep external data disabled unless your model is huge.
        use_external_data_format=False,

        # Keep the whole OBB head (model.22) in FP32 — class, box/DFL and angle
        # branches. Quantizing only the class tail left the box/angle regression
        # int8, which crushed IoU on rotated boxes (railroad-crossing, trefolo).
        nodes_to_exclude=nodes_to_exclude,
    )

    print("Done.")