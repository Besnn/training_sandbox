"""Convert a trained FOMO v3 checkpoint to a static INT8 ONNX model.

FOMO v3 uses the full-width MobileNetV2 (alpha=1.0) backbone cut at block 13
with a depthwise-3x3 -> ReLU -> 1x1 -> ReLU -> 1x1 head (see
convert_fomo_v3_to_onnx.py / FOMO_V3_480). num_classes is auto-detected from
the checkpoint, so no data.yaml is required.

Usage:
    python convert_fomo_v3_to_int8.py
    python convert_fomo_v3_to_int8.py --checkpoint models/fomo-pt/fomo_v3_480.pt \
                                       --onnx-int8 models/fomo-480-onnx/fomo-v3-480-int8.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)
from PIL import Image

from convert_fomo_v3_to_onnx import FOMO_V3_480


DEFAULT_MODEL_IN = "models/fomo-pt/fomo_v3_480.pt"
DEFAULT_ONNX_FP32 = "models/fomo-480-onnx/fomo-v3-480.onnx"
DEFAULT_ONNX_PREPROCESSED = "models/tmp/fomo-v3-onnx-quant-preprocessed.onnx"
DEFAULT_ONNX_INT8 = "models/fomo-480-onnx/fomo-v3-480-int8.onnx"
DEFAULT_CALIB_DIR = "datasets/split_centroid_dataset/train/images"


class FOMOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, img_size=480, max_images=100):
        image_dir = Path(image_dir)
        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.image_paths.extend(image_dir.glob(ext))

        self.image_paths = sorted(self.image_paths)[:max_images]
        self.input_name = input_name
        self.img_size = img_size
        self.index = 0

        if not self.image_paths:
            raise RuntimeError(f"No calibration images found in {image_dir}")

    def preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_array = image_array.transpose(2, 0, 1)  # HWC -> CHW
        image_array = np.expand_dims(image_array, axis=0)

        return image_array

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        return {self.input_name: self.preprocess(image_path)}


def export_fomo_v3_to_onnx(checkpoint_path, output_path, img_size, opset):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = FOMO_V3_480.from_checkpoint(checkpoint_path)
    model.eval()

    dummy_input = torch.zeros(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        export_params=True,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    )


def get_input_name(model_path):
    model = onnx.load(model_path)
    return model.graph.input[0].name


def convert_fomo_v3_to_int8(args):
    print(f"Exporting FP32 ONNX: {args.checkpoint} -> {args.onnx_fp32}")
    export_fomo_v3_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.onnx_fp32,
        img_size=args.img_size,
        opset=args.opset,
    )

    print(f"Preprocessing ONNX for quantization: {args.onnx_fp32} -> {args.onnx_preprocessed}")
    Path(args.onnx_preprocessed).parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(
        input_model_path=args.onnx_fp32,
        output_model_path=args.onnx_preprocessed,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    input_name = get_input_name(args.onnx_preprocessed)
    calibration_reader = FOMOCalibrationDataReader(
        image_dir=args.calib_dir,
        input_name=input_name,
        img_size=args.img_size,
        max_images=args.num_calib_images,
    )

    Path(args.onnx_int8).parent.mkdir(parents=True, exist_ok=True)
    print(f"Quantizing INT8 ONNX: {args.onnx_preprocessed} -> {args.onnx_int8}")
    quantize_static(
        model_input=args.onnx_preprocessed,
        model_output=args.onnx_int8,
        calibration_data_reader=calibration_reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv"],
        per_channel=True,
        use_external_data_format=False,
    )

    print(f"Done: {args.onnx_int8}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the FOMO v3 PyTorch model to INT8 ONNX.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_IN, help="Input fomo_v3_480.pt checkpoint.")
    parser.add_argument("--onnx-fp32", default=DEFAULT_ONNX_FP32, help="Intermediate FP32 ONNX path.")
    parser.add_argument(
        "--onnx-preprocessed",
        default=DEFAULT_ONNX_PREPROCESSED,
        help="Intermediate ONNX path after quantization preprocessing.",
    )
    parser.add_argument("--onnx-int8", default=DEFAULT_ONNX_INT8, help="Output INT8 ONNX path.")
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR, help="Calibration image directory.")
    parser.add_argument("--img-size", type=int, default=480, help="Square FOMO input size.")
    parser.add_argument("--num-calib-images", type=int, default=100, help="Number of calibration images.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")

    return parser.parse_args()


if __name__ == "__main__":
    convert_fomo_v3_to_int8(parse_args())
