"""Convert a trained FOMO v3 checkpoint to ONNX (fp32).

FOMO v3 uses full-width MobileNetV2 (alpha=1.0) cut at block 13 (stride 16,
96 ch) with a 3-layer head: depthwise 3x3 -> ReLU -> 1x1 96->96 -> ReLU ->
1x1 96->num_classes. This is distinct from the FOMO_PL_480 thin head
(1x1 -> ReLU -> 1x1) in training_fomo_v2.py, so it has its own script.

Usage:
    python convert_fomo_v3_to_onnx.py
    python convert_fomo_v3_to_onnx.py --model-in path/to/fomo_v3_480.pt \
                                       --model-out path/to/fomo-v3-480.onnx \
                                       --num-classes 4
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

DEFAULT_MODEL_IN  = "models/fomo-pt/fomo_v5_480.pt"
DEFAULT_MODEL_OUT = ("models/fomo-480-onnx/fomo-v5-480.onnx")


# ---------------------------------------------------------------------------
# Architecture (mirrors FOMO_V3_480 in the training notebook)
# ---------------------------------------------------------------------------

class FOMO_V3_480(nn.Module):
    """STN-FOMO architecture without the Spatial Transformer.

    Full-width MobileNetV2 (alpha=1.0) cut at block 13 -> stride 16
    (480 -> 30x30, 96 ch). Head: depthwise 3x3 -> ReLU -> 1x1 96->96 ->
    ReLU -> 1x1 96->num_classes. Returns raw logits (sigmoid at decode).
    """

    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None, width_mult=1.0).features
        self.features = nn.Sequential(*list(backbone.children())[:14])
        backbone_channels = self._infer_backbone_channels()
        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3,
                      padding=1, groups=backbone_channels),
            nn.ReLU(),
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(backbone_channels, num_classes, kernel_size=1),
        )

    def _infer_backbone_channels(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            ch = self.features(torch.zeros(1, 3, 64, 64)).shape[1]
        self.train(was_training)
        return ch

    def forward(self, x):
        return self.head(self.features(x))

    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location="cpu"):
        """Load a saved v3 checkpoint. Infers num_classes from head.4.weight."""
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        num_classes = state_dict["head.4.weight"].shape[0]
        model = cls(num_classes=num_classes)
        # strict=False: ignores loc_fc.* keys present in checkpoints trained
        # with the localization auxiliary branch (training-only, not exported).
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if not k.startswith("loc_fc.")]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch — missing: {missing}, unexpected: {unexpected}"
            )
        return model


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export FOMO v3 to ONNX (fp32)")
    parser.add_argument("--model-in",    default=DEFAULT_MODEL_IN,
                        help="Path to fomo_v3_480.pt checkpoint")
    parser.add_argument("--model-out",   default=DEFAULT_MODEL_OUT,
                        help="Destination .onnx path")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Override num_classes (auto-detected from checkpoint)")
    parser.add_argument("--opset",       type=int, default=17,
                        help="ONNX opset version (default 17)")
    args = parser.parse_args()

    model_in  = Path(args.model_in)
    model_out = Path(args.model_out)

    if not model_in.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_in}")

    model_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {model_in}")
    model = FOMO_V3_480.from_checkpoint(model_in)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    out_shape = tuple(model(torch.zeros(1, 3, 480, 480)).shape)
    print(f"  parameters : {n_params:,}")
    print(f"  output shape: {out_shape}")

    dummy = torch.zeros(1, 3, 480, 480)
    print(f"Exporting to: {model_out}  (opset {args.opset})")
    torch.onnx.export(
        model,
        dummy,
        str(model_out),
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
    )
    print("Done.")


if __name__ == "__main__":
    main()
