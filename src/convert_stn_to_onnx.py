"""Export the trained STN-FOMO model (train_stn_fomo.py) to ONNX.

The STN head warps the feature map with F.grid_sample; that exports to ONNX as
the GridSample op (opset >= 16), so opset 21 here is fine. num_classes is detected
from the checkpoint's final head conv, so this works regardless of class count.
"""

import torch
from pathlib import Path

from train_stn_fomo import STN_FOMO_480

MODEL_IN = "stn_fomo_mac.pt"
MODEL_OUT = "models/stn-fomo-480-onnx/stn-fomo-480.onnx"
IMG_SIZE = 480


def detect_num_classes(state_dict):
    """num_classes = output channels of the final 1x1 classifier conv in the head."""
    head_convs = [v for k, v in state_dict.items() if k.startswith("head.") and v.ndim == 4]
    if not head_convs:
        raise ValueError("Could not find a head conv in the checkpoint to infer num_classes.")
    return head_convs[-1].shape[0]


def main():
    Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(MODEL_IN, map_location="cpu")
    num_classes = detect_num_classes(state_dict)

    model = STN_FOMO_480(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_OUT,
        opset_version=21,
        export_params=True,
        external_data=False,
        input_names=["images"],
        output_names=["output"],
    )

    print(f"Esportazione completata in {MODEL_OUT} (num_classes={num_classes})")


if __name__ == "__main__":
    main()
