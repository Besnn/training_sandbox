"""Convert the trained STN-FOMO PyTorch model to INT8 ONNX.

This mirrors convert_fomo_to_int8.py, but uses the STN-FOMO architecture from
train_stn_fomo.py and infers the number of classes from the checkpoint's final
head convolution. The STN GridSample op is exported in the FP32 graph and ONNX
Runtime quantization will quantize supported surrounding ops.
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

from train_stn_fomo import STN_FOMO_480


DEFAULT_MODEL_IN = "stn_fomo_mac.pt"
DEFAULT_ONNX_FP32 = "models/stn-fomo-480-onnx/stn-fomo-480.onnx"
DEFAULT_ONNX_PREPROCESSED = "models/stn-fomo-onnx-quant-preprocessed.onnx"
DEFAULT_ONNX_INT8 = "models/stn-fomo-480-onnx/stn-fomo-480-int8.onnx"
DEFAULT_CALIB_DIR = "datasets/split_centroid_dataset/train/images"


class STNCalibrationDataReader(CalibrationDataReader):
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


def detect_num_classes(state_dict):
    """num_classes = output channels of the final 1x1 classifier conv in the head."""
    head_convs = [v for k, v in state_dict.items() if k.startswith("head.") and v.ndim == 4]
    if not head_convs:
        raise ValueError("Could not find a head conv in the checkpoint to infer num_classes.")
    return head_convs[-1].shape[0]


def export_stn_to_onnx(checkpoint_path, output_path, img_size, opset):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    num_classes = detect_num_classes(state_dict)

    model = STN_FOMO_480(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        export_params=True,
        external_data=False,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes=None,
    )

    return num_classes


def get_input_name(model_path):
    model = onnx.load(model_path)
    return model.graph.input[0].name


def convert_stn_to_int8(args):
    print(f"Exporting FP32 ONNX: {args.checkpoint} -> {args.onnx_fp32}")
    num_classes = export_stn_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.onnx_fp32,
        img_size=args.img_size,
        opset=args.opset,
    )
    print(f"Classes: {num_classes}")

    print(f"Preprocessing ONNX for quantization: {args.onnx_fp32} -> {args.onnx_preprocessed}")
    quant_pre_process(
        input_model_path=args.onnx_fp32,
        output_model_path=args.onnx_preprocessed,
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    input_name = get_input_name(args.onnx_preprocessed)
    calibration_reader = STNCalibrationDataReader(
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
        per_channel=True,
        use_external_data_format=False,
    )

    print(f"Done: {args.onnx_int8}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the STN-FOMO PyTorch model to INT8 ONNX.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_IN, help="Input STN-FOMO .pt state dict.")
    parser.add_argument("--onnx-fp32", default=DEFAULT_ONNX_FP32, help="Intermediate FP32 ONNX path.")
    parser.add_argument(
        "--onnx-preprocessed",
        default=DEFAULT_ONNX_PREPROCESSED,
        help="Intermediate ONNX path after quantization preprocessing.",
    )
    parser.add_argument("--onnx-int8", default=DEFAULT_ONNX_INT8, help="Output INT8 ONNX path.")
    parser.add_argument("--calib-dir", default=DEFAULT_CALIB_DIR, help="Calibration image directory.")
    parser.add_argument("--img-size", type=int, default=480, help="Square STN-FOMO input size.")
    parser.add_argument("--num-calib-images", type=int, default=100, help="Number of calibration images.")
    parser.add_argument("--opset", type=int, default=21, help="ONNX opset version.")

    return parser.parse_args()


if __name__ == "__main__":
    convert_stn_to_int8(parse_args())
