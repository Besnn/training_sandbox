"""Measure the STN localization net output (cos, sin) and derived scale/angle
across 30 images from the validation set, to characterise what the Spatial
Transformer is actually doing at inference.

The STN's loc_fc predicts (cos θ, sin θ). Because the grid is NOT renormalized
to the unit circle, the magnitude sqrt(cos²+sin²) gives the actual scale applied
by GridSample. A pure rotation would give magnitude=1.0; a value of ~0.961 means
the STN is applying a ~4% uniform zoom-in with negligible rotation.

Usage:
    python measure_stn_scale.py
    python measure_stn_scale.py --model path/to/stn-fomo.onnx \
                                --images path/to/val/images \
                                --n 30
"""

import argparse
import glob
import math
import os
import statistics

import cv2
import numpy as np
import onnx
from onnx import helper, shape_inference
import onnxruntime as ort

# --- defaults ----------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
ROOT     = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_MODEL  = os.path.join(HERE, "models", "stn-fomo-480-onnx", "stn-fomo-480.onnx")
DEFAULT_IMAGES = os.path.join(HERE, "datasets", "yolo_pl_test", "images")
DEFAULT_N      = 30


# --- helpers -----------------------------------------------------------------
def build_session(model_path):
    """Load the ONNX and expose the loc_fc output ('linear_1') as an extra output."""
    m = onnx.load(model_path)
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass

    vmap    = {vi.name: vi for vi in m.graph.value_info}
    existing = {o.name for o in m.graph.output}

    if "linear_1" not in existing:
        if "linear_1" in vmap:
            m.graph.output.append(vmap["linear_1"])
        else:
            m.graph.output.append(helper.make_empty_tensor_value_info("linear_1"))

    return ort.InferenceSession(
        m.SerializeToString(), providers=["CPUExecutionProvider"]
    )


def preprocess(path):
    bgr  = cv2.imread(path)
    img  = cv2.resize(bgr, (480, 480))
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return blob


def measure(sess, img_path):
    blob     = preprocess(img_path)
    in_name  = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    results  = dict(zip(out_names, sess.run(out_names, {in_name: blob})))
    cos_v, sin_v = float(results["linear_1"][0, 0]), float(results["linear_1"][0, 1])
    scale = math.sqrt(cos_v**2 + sin_v**2)
    angle = math.degrees(math.atan2(sin_v, cos_v))
    return cos_v, sin_v, scale, angle


# --- main --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Measure STN loc_fc scale/angle across N val images")
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help="STN-FOMO fp32 ONNX path")
    parser.add_argument("--images", default=DEFAULT_IMAGES, help="Directory of val images")
    parser.add_argument("--n",      default=DEFAULT_N, type=int, help="Number of images to sample")
    args = parser.parse_args()

    paths = sorted(
        p for p in glob.glob(os.path.join(args.images, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {args.images}")

    paths = paths[:args.n]
    print(f"Model : {args.model}")
    print(f"Images: {args.images}  ({len(paths)} of {args.n} requested)")
    print()

    sess = build_session(args.model)

    scales, angles = [], []

    print(f"{'image':40s}  {'cos':>7}  {'sin':>8}  {'scale':>7}  {'angle°':>8}")
    print("─" * 78)

    for p in paths:
        cos_v, sin_v, scale, angle = measure(sess, p)
        scales.append(scale)
        angles.append(angle)
        name = os.path.basename(p)
        print(f"{name:40s}  {cos_v:>7.4f}  {sin_v:>8.4f}  {scale:>7.4f}  {angle:>+8.3f}°")

    print("─" * 78)
    print(f"{'mean':40s}  {'':>7}  {'':>8}  {statistics.mean(scales):>7.4f}  {statistics.mean(angles):>+8.3f}°")
    print(f"{'std':40s}  {'':>7}  {'':>8}  {statistics.stdev(scales):>7.4f}  {statistics.stdev(angles):>+8.3f}°")
    print(f"{'min':40s}  {'':>7}  {'':>8}  {min(scales):>7.4f}  {min(angles):>+8.3f}°")
    print(f"{'max':40s}  {'':>7}  {'':>8}  {max(scales):>7.4f}  {max(angles):>+8.3f}°")
    print()
    print(f"Interpretation:")
    print(f"  Mean scale  {statistics.mean(scales):.4f}  → GridSample samples from "
          f"±{statistics.mean(scales):.4f} instead of ±1.0")
    print(f"  Mean angle  {statistics.mean(angles):+.3f}°  → rotation applied by the STN")
    print(f"  Scale std   {statistics.stdev(scales):.4f}  → {'nearly constant (not input-dependent)' if statistics.stdev(scales) < 0.005 else 'varies with input'}")


if __name__ == "__main__":
    main()
