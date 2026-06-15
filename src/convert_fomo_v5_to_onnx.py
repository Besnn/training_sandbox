"""Convert a trained FOMO v5 (blur-bottleneck) checkpoint to ONNX (fp32).

FOMO v5 is the v3/v4 architecture (full-width MobileNetV2 alpha=1.0 cut at
block 13 -> 96ch @ 30x30, depthwise+pointwise head) PLUS a fixed, non-learnable
`grid_sample` blur (constant ~0.967x scale, zero rotation) wired between the
backbone and the head -- present in BOTH training and inference. It has NO
loc_fc localization branch (that auxiliary path lives only in v4).

This needs its own conversion script for two reasons:
  1. `convert_fomo_v3_to_onnx.py` defaults to `fomo_v4_480.pt` -- pointing it at
     the v5 checkpoint without also fixing the architecture would silently
     export v4's weights again (or drop the blur as an "unexpected key").
  2. The `FOMO_V3_480` class there has no `blur` submodule, so
     `blur.base_grid` would be filtered out by the `strict=False` key handling,
     producing an ONNX graph with NO `GridSample` node -- i.e. functionally a
     v3/v4 model wearing a v5 filename. (This is exactly what happened the
     first time `fomo-v5-480.onnx` was produced: bit-for-bit identical to
     `fomo-v4-480.onnx`, max|diff| = 0.0 across 10 test images, no GridSample
     in the graph.)

Usage:
    python convert_fomo_v5_to_onnx.py
    python convert_fomo_v5_to_onnx.py --model-in path/to/fomo_v5_480.pt \
                                       --model-out path/to/fomo-v5-480.onnx \
                                       --num-classes 4
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DEFAULT_MODEL_IN  = "models/fomo-pt/fomo_v5_480.pt"
DEFAULT_MODEL_OUT = "models/fomo-480-onnx/fomo-v5-480.onnx"

# Must match the value baked into the trained checkpoint's `blur.base_grid`
# buffer (set in the v5 training notebook). Re-derived at load time below as
# a sanity check -- this constant is only the fallback / documentation value.
BLUR_SCALE = 0.967


# ---------------------------------------------------------------------------
# Architecture (mirrors FOMO_V5_480 in the v5 training notebook)
# ---------------------------------------------------------------------------

class FixedGridSampleBlur(nn.Module):
    """Fixed, non-learnable bilinear resampling bottleneck.

    Constant affine transform (uniform scale, zero rotation) applied via
    bilinear grid_sample on the 30x30 feature grid. No learnable parameters --
    `base_grid` is a registered buffer, so it round-trips through state_dict
    and the export, and shows up in the ONNX graph as a single `GridSample`
    node with a constant `grid` input.
    """

    def __init__(self, scale=BLUR_SCALE, grid_size=30):
        super().__init__()
        theta = torch.tensor([[[scale, 0.0, 0.0],
                               [0.0, scale, 0.0]]], dtype=torch.float32)
        grid = F.affine_grid(theta, (1, 1, grid_size, grid_size), align_corners=False)
        self.register_buffer("base_grid", grid)

    def forward(self, x):
        b = x.shape[0]
        grid = self.base_grid.expand(b, -1, -1, -1)
        return F.grid_sample(x, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)


class FOMO_V5_480(nn.Module):
    """v3/v4 architecture + fixed blur bottleneck, NO localization branch.

    Full-width MobileNetV2 (alpha=1.0) cut at block 13 -> stride 16 (480 -> 30,
    96 ch) -> FixedGridSampleBlur -> head: depthwise 3x3 -> ReLU -> 1x1 96->96
    -> ReLU -> 1x1 96->num_classes. Returns raw logits. The forward graph is
    identical in train and eval -- the blur is always on, exactly like the STN
    in STN-FOMO (the head is calibrated to the blurred features).
    """

    def __init__(self, num_classes, blur_scale=BLUR_SCALE):
        super().__init__()
        backbone = models.mobilenet_v2(weights=None, width_mult=1.0).features
        self.features = nn.Sequential(*list(backbone.children())[:14])
        backbone_channels = self._infer_backbone_channels()
        self.blur = FixedGridSampleBlur(scale=blur_scale, grid_size=30)
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
        feat = self.features(x)
        feat = self.blur(feat)
        return self.head(feat)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, map_location="cpu"):
        """Load a saved v5 checkpoint. Infers num_classes from head.4.weight and
        the actual blur scale from the checkpoint's `blur.base_grid` buffer
        (rather than trusting BLUR_SCALE), so the exported graph exactly matches
        what the model was trained with."""
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if "blur.base_grid" not in state_dict:
            raise RuntimeError(
                "Checkpoint has no 'blur.base_grid' buffer -- this looks like a "
                "v3/v4 checkpoint (no fixed blur), not a v5 one. Did you point "
                "--model-in at the wrong .pt file?"
            )

        num_classes = state_dict["head.4.weight"].shape[0]
        scale = cls._scale_from_grid(state_dict["blur.base_grid"])
        print(f"  detected blur scale in checkpoint: {scale:.5f}  "
              f"(module default: {BLUR_SCALE})")

        model = cls(num_classes=num_classes, blur_scale=scale)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # base_grid is a buffer we just (re)derived analytically above; allow the
        # loader to skip re-assigning it (it's numerically identical either way).
        missing = [k for k in missing if k != "blur.base_grid"]
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch — missing: {missing}, unexpected: {unexpected}"
            )
        return model

    @staticmethod
    def _scale_from_grid(grid):
        """Recover the uniform affine scale baked into an affine_grid buffer.

        affine_grid with theta=[[s,0,0],[0,s,0]] produces, at align_corners=False,
        a grid whose values are an affine function of normalized pixel position
        with slope `s`. The corner-to-corner span of either axis equals
        `s * (1 - 1/grid_size)`, which we invert to recover `s`.
        """
        gs = grid.shape[1]  # (1, gs, gs, 2)
        xs = grid[0, gs // 2, :, 0]
        span = float(xs[-1] - xs[0])
        return span / (2.0 * (1.0 - 1.0 / gs))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export FOMO v5 (blur-bottleneck) to ONNX (fp32)")
    parser.add_argument("--model-in",    default=DEFAULT_MODEL_IN,
                        help="Path to fomo_v5_480.pt checkpoint")
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
    model = FOMO_V5_480.from_checkpoint(model_in)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    out_shape = tuple(model(torch.zeros(1, 3, 480, 480)).shape)
    print(f"  parameters : {n_params:,}  (blur adds 0 — it's a fixed buffer)")
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

    # Sanity check: confirm the GridSample survived export (this is the whole
    # point -- the first attempt at this conversion silently dropped it).
    import onnx
    om = onnx.load(str(model_out))
    gs_nodes = [n.name for n in om.graph.node if n.op_type == "GridSample"]
    if not gs_nodes:
        raise RuntimeError(
            "Exported graph has NO GridSample node -- the blur was dropped "
            "during export. This is the exact bug this script exists to avoid; "
            "something is wrong with the model definition above."
        )
    print(f"  GridSample node present: {gs_nodes}  (blur survived export ✓)")
    print("Done.")


if __name__ == "__main__":
    main()
