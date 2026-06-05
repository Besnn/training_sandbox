import argparse
from pathlib import Path

import onnx
import torch
import yaml

from training_fomo_v2 import FOMO_PL_480


DEFAULT_MODEL_IN = "fomo_pl_mac.pt"
DEFAULT_MODEL_OUT = "models/fomo_fp16.onnx"
DEFAULT_DATA_YAML = "datasets/split_centroid_dataset/data.yaml"


def load_num_classes(data_yaml):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg.get("names")
    if isinstance(names, dict):
        return len(names)
    if isinstance(names, list):
        return len(names)

    raise ValueError(f"Could not read class names from {data_yaml}")


def export_fomo_to_fp16_onnx(checkpoint_path, output_path, num_classes, img_size, opset):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Auto-detects the backbone width (alpha 1.0 / 0.35) from the checkpoint.
    model = FOMO_PL_480.from_checkpoint(checkpoint_path, num_classes=num_classes)
    model.eval()
    model.half()

    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float16)

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

    onnx.checker.check_model(onnx.load(output_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert the FOMO PyTorch model to FP16 ONNX.")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL_IN, help="Input FOMO .pt state dict.")
    parser.add_argument("--onnx-out", default=DEFAULT_MODEL_OUT, help="Output FP16 ONNX path.")
    parser.add_argument("--data-yaml", default=DEFAULT_DATA_YAML, help="Dataset YAML used to infer classes.")
    parser.add_argument("--img-size", type=int, default=480, help="Square FOMO input size.")
    parser.add_argument("--opset", type=int, default=21, help="ONNX opset version.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    classes = load_num_classes(args.data_yaml)

    print(f"Classes: {classes}")
    print(f"Exporting FP16 ONNX: {args.checkpoint} -> {args.onnx_out}")
    export_fomo_to_fp16_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.onnx_out,
        num_classes=classes,
        img_size=args.img_size,
        opset=args.opset,
    )
    print(f"Done: {args.onnx_out}")
