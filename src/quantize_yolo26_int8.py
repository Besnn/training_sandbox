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

MODEL_FP32 = "models/tmp/yolo26n-onnx-quant-preprocessed.onnx"
MODEL_INT8 = "models/yolo26n-obb-onnx/yolo26n-obb-int8.onnx"
CALIB_DIR = "datasets/split_obb_dataset/train/images"

IMG_SIZE = 640
NUM_CALIB_IMAGES = 150

# The OBB head in YOLO26 is module model.23 (it is model.22 in YOLOv8). The
# Keep the ENTIRE OBB detect head (module model.23) in FP32. Quantizing it
# does two kinds of damage:
#   - the class branch shares the final Concat's uint8 scale with box coords
#     (~660) and angle (~pi), which crushes class scores to zero; and
#   - the box-regression (cv2 / DFL) and angle (cv4) branches lose coordinate
#     precision, which tanks IoU for long, thin, rotated boxes (railroad-
#     crossing, trefolo) while compact classes (lights-on/off) shrug it off.
# The backbone + neck (where almost all the size/latency win lives) stay int8.
# NOTE: it is model.23 in YOLO26 (model.22 in YOLOv8), and quantize_static
# silently ignores unmatched names — the guard below catches a wrong prefix.
HEAD_PREFIX = "/model.23/"


def head_nodes(model_path, prefix=HEAD_PREFIX):
    graph = onnx.load(model_path).graph
    return [n.name for n in graph.node if n.name.startswith(prefix)]


NODES_TO_EXCLUDE = head_nodes(MODEL_FP32)


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
    """Fail loudly if an exclude name does not match any node in the graph.

    quantize_static ignores unmatched names, which would quantize the OBB head
    and produce an all-zero (no-detection) int8 model.
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

    print(f"Input name: {input_name}")
    print(f"Quantizing: {MODEL_FP32} -> {MODEL_INT8}")

    check_exclude_nodes(MODEL_FP32, NODES_TO_EXCLUDE)

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

        # Quantize ONLY Conv. By default quantize_static also quantizes MatMul,
        # and YOLO26's attention MatMuls (model.10 / model.22 PSA blocks) then
        # fuse to QLinearMatMul(21) at session load — a kernel the Arduino Uno Q
        # onnxruntime build does not implement, so the session fails with
        # NOT_IMPLEMENTED. Conv is the bulk of the compute anyway, and keeping
        # attention FP32 is better for accuracy.
        op_types_to_quantize=["Conv"],

        # Keep external data disabled unless your model is huge.
        use_external_data_format=False,

        # Keep the OBB head (model.23) tail in FP32: final Concat plus the
        # class-score and angle branches. A single shared uint8 scale across
        # box coords / class sigmoids / angle collapses class scores to zero.
        nodes_to_exclude=NODES_TO_EXCLUDE,
    )

    print("Done.")
