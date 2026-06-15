"""
FOMO / STN-FOMO feature-map inspector.

Supports both model families (selectable in the UI):
  fomo     : MobileNetV2 α=0.35, stride 8, 32ch backbone, 60×60 grid,
             5-channel output (background + 4 classes, softmax).
  stn-fomo : MobileNetV2 α=1.0,  stride 16, 96ch backbone, 30×30 grid,
             4-channel output (per-class sigmoid, no background).

Each family has its own selectable intermediate layers. Both accept a
float NCHW [1,3,480,480] input scaled to 0..1 and come in fp32 / int8
variants.
"""

import os
import base64

import cv2
import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request

# --- paths -----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
_SANDBOX = os.path.join(PROJECT_ROOT, "traffic_signal_detection", "src" ,"models")

MODELS = {
    "stn-fomo": {
        "fp32": os.path.join(_SANDBOX, "stn-fomo-480-onnx", "stn-fomo-480.onnx"),
        "int8": os.path.join(_SANDBOX, "stn-fomo-480-onnx", "stn-fomo-480-int8.onnx"),
    },
    "fomo": {
        "fp32": os.path.join(_SANDBOX, "fomo-480-onnx", "fomo-480.onnx"),
        "int8": os.path.join(_SANDBOX, "fomo-480-onnx", "fomo-480-int8.onnx"),
    },
    "fomo-v4": {
        "fp32": os.path.join(PROJECT_ROOT, "traffic_signal_detection", "src",
                             "models", "fomo-480-onnx", "fomo-v3-480.onnx"),
    },
}

# Intermediate layers selectable per model family.
# Order matters: the first entry is the default shown by the UI.
LAYERS_BY_FAMILY = {
    "stn-fomo": {
        "add_7":       "add_7 — MobileNetV2 backbone output (BEFORE rotation / STN input)",
        "grid_sampler":"grid_sampler — STN output (AFTER rotation)",
        "relu_1":      "relu_1 — head conv (96→96) post-ReLU",
        "relu_2":      "relu_2 — final 96 feature maps before detect head",
    },
    "fomo": {
        "relu":      "relu — final 32 feature maps before classifier",
        "conv2d_18": "conv2d_18 — head conv (32→32) pre-ReLU",
        "add_2":     "add_2 — backbone output, last residual block (32ch)",
    },
    "fomo-v4": {
        "add_900":   "add_900 — MobileNetV2 backbone output (96ch, 30×30)",
        "conv2d_39": "conv2d_39 — head depthwise 3×3 pre-ReLU",
        "relu":      "relu — head depthwise 3×3 post-ReLU",
        "conv2d_40": "conv2d_40 — head pointwise 96→96 pre-ReLU",
        "relu_1":    "relu_1 — final 96 feature maps before detect head",
    },
}

# Per-family constants
MODEL_INFO = {
    "stn-fomo": {"num_maps": 96, "grid": 30, "bg_idx": None},
    "fomo":     {"num_maps": 32, "grid": 60, "bg_idx": 0},
    "fomo-v4":  {"num_maps": 96, "grid": 30, "bg_idx": None},
}

CLASSES = ["railroad-crossing", "lights-on", "lights-off", "trefolo"]
IN_W = IN_H = 480

# Families that have a "before warp" / "after warp" layer pair -- currently
# only stn-fomo: add_7 is the raw MobileNetV2 backbone output fed INTO the STN,
# grid_sampler is what comes OUT of it (after the learned rotation/scale is
# applied). When present, both are exposed as extra graph outputs so the
# "most affected by the warp" section can be computed from the SAME forward
# pass that produces everything else -- no second inference run needed.
ROTATION_DIFF_LAYERS = {
    "stn-fomo": ("add_7", "grid_sampler"),
}
ROTATION_DIFF_TOP_N = 12

# The actual sampling-grid tensor fed into each family's GridSample node (the
# second input -- ['add_7', <this name>] -> 'grid_sampler'). It's the SAME
# tensor name in fp32 and int8 graphs (`stack`), even though add_7/grid_sampler
# get DequantizeLinear aliases internally in the int8 graph -- onnxruntime
# still resolves `stack` to a float32 (H, W, 2) tensor in both. This is what
# lets us reconstruct the STN's *actual, per-image* warp displacement field.
ROTATION_DIFF_GRID_LAYER = {
    "stn-fomo": "stack",
}

COLORMAPS = {
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "jet":     cv2.COLORMAP_JET,
    "magma":   cv2.COLORMAP_MAGMA,
    "gray":    None,
}

app = Flask(__name__)

_SESSIONS = {}  # keyed by (family, precision, layer_name, bypass)


def get_session(family, precision, layer_name, bypass=False):
    key = (family, precision, layer_name, bypass)
    if key in _SESSIONS:
        return _SESSIONS[key]

    model = onnx.load(MODELS[family][precision])
    try:
        model = shape_inference.infer_shapes(model)
    except Exception:
        pass

    vmap = {vi.name: vi for vi in model.graph.value_info}
    existing = {o.name for o in model.graph.output}

    wanted = {layer_name}
    wanted |= set(ROTATION_DIFF_LAYERS.get(family, ()))
    grid_layer_name = ROTATION_DIFF_GRID_LAYER.get(family)
    if grid_layer_name:
        wanted.add(grid_layer_name)

    for name in wanted:
        if name in existing:
            continue
        if name in vmap:
            model.graph.output.append(vmap[name])
        else:
            model.graph.output.append(helper.make_empty_tensor_value_info(name))

    if bypass:
        _patch_bypass_stn(model)

    sess = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    _SESSIONS[key] = sess
    return sess


def preprocess(bgr):
    img = cv2.resize(bgr, (IN_W, IN_H))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    return blob, rgb


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def png_b64(bgr):
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_tile(channel, colormap, tile_px, vmin, vmax):
    span = vmax - vmin
    if span <= 1e-9:
        norm = np.zeros_like(channel)
    else:
        norm = np.clip((channel - vmin) / span, 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    u8 = cv2.resize(u8, (tile_px, tile_px), interpolation=cv2.INTER_NEAREST)
    cmap = COLORMAPS.get(colormap, cv2.COLORMAP_VIRIDIS)
    if cmap is None:
        out = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    else:
        out = cv2.applyColorMap(u8, cmap)
    return png_b64(out)


_IDENTITY_GRID_CACHE = {}


def _identity_grid(size):
    """The (size, size, 2) sampling grid `F.affine_grid` produces for the
    IDENTITY transform (scale=1, rotation=0) at align_corners=False -- a fixed
    function of grid size alone, the same for every image. Per-pixel-center
    normalized coordinates are `(2*i + 1) / size - 1` along each axis; the
    last dim is (x, y) to match grid_sample's convention."""
    if size not in _IDENTITY_GRID_CACHE:
        coords = (2.0 * np.arange(size, dtype=np.float64) + 1.0) / size - 1.0
        xs, ys = np.meshgrid(coords, coords)  # 'xy' indexing: xs along width, ys along height
        _IDENTITY_GRID_CACHE[size] = np.stack([xs, ys], axis=-1)
    return _IDENTITY_GRID_CACHE[size]


def _patch_bypass_stn(model):
    """Replace the STN's learned sampling grid with a fixed identity grid,
    effectively bypassing the spatial transformer. The backbone output (add_7)
    flows through grid_sample unchanged — equivalent to θ = [[1,0,0],[0,1,0]].

    Works for both fp32 and int8 graphs: we overwrite the second input of the
    GridSample node (always float32 in both variants) with a Constant node
    that holds the identity grid. The localization network still runs and its
    output tensor ('stack') is still exposed, so the UI can still show *what
    the STN wanted to do* even though it was bypassed.

    Returns True if the graph was patched, False if no GridSample was found.
    """
    gs_node = next((n for n in model.graph.node if n.op_type == "GridSample"), None)
    if gs_node is None:
        return False

    # Detect the grid's spatial size from shape info; default to 30 (stn-fomo).
    H = W = 30
    grid_input_name = gs_node.input[1]
    vmap = {vi.name: vi for vi in model.graph.value_info}
    if grid_input_name in vmap:
        shape = vmap[grid_input_name].type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            h_val = shape.dim[1].dim_value
            w_val = shape.dim[2].dim_value
            if h_val > 0 and w_val > 0:
                H, W = h_val, w_val

    # Identity grid (1, H, W, 2) — align_corners=False pixel-centre coordinates.
    cols = (2.0 * np.arange(W, dtype=np.float32) + 1.0) / W - 1.0
    rows = (2.0 * np.arange(H, dtype=np.float32) + 1.0) / H - 1.0
    xs, ys = np.meshgrid(cols, rows)
    identity = np.stack([xs, ys], axis=-1)[np.newaxis]  # (1, H, W, 2)

    const_name = "__bypass_identity_grid__"
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=[const_name],
        value=numpy_helper.from_array(identity, name=const_name),
    )
    gs_node.input[1] = const_name
    model.graph.node.insert(0, const_node)
    return True


def displacement_field(grid):
    """Per-output-pixel warp displacement magnitude |δ(x)|, reconstructed from
    the STN's ACTUAL (per-image, learned) sampling grid.

    `grid` is the (H, W, 2) tensor the GridSample node used for this image
    (normalized [-1, 1] coords, align_corners=False). Subtracting the fixed
    identity grid isolates exactly what the STN's predicted affine transform
    displaced each location by, in normalized coordinates; multiplying by
    size/2 converts that to OUTPUT-PIXEL units -- the same units a
    finite-difference image gradient is expressed in, so
    `|∇before(x)| * |δ(x)|` is dimensionally a first-order (Taylor) estimate
    of how much the value at x should change under the warp.
    """
    size = grid.shape[0]
    delta_norm = grid.astype(np.float64) - _identity_grid(size)
    return np.sqrt((delta_norm ** 2).sum(axis=-1)) * (size / 2.0)


def _safe_corr(a, b):
    if a.std() > 1e-12 and b.std() > 1e-12:
        return float(np.corrcoef(a, b)[0, 1])
    return None


def build_rotation_diff(before, after, colormap, tile_px, grid=None, top_n=ROTATION_DIFF_TOP_N):
    """Rank channels by how much the STN's warp changed them, and render
    before / after / |Δ| / predicted-|Δ| tiles for the top-N most-changed ones.

    `before` and `after` are (C, H, W) arrays from the SAME forward pass --
    add_7 (pre-STN) and grid_sampler (post-STN) respectively. Channels are
    ranked by mean absolute per-pixel difference. before/after tiles share a
    single colour-scale (the union of their value ranges) so the visual
    comparison is apples-to-apples -- a channel the warp left untouched will
    render pixel-identical; the |Δ| tile then highlights exactly where (and
    by how much) the two diverge, on its own 0..max scale.

    Hypothesis check -- "small-warp / first-order (Taylor)" model: STN-FOMO's
    learned transform is a small, near-identity affine perturbation (measured
    scale ~0.96-0.97, rotation a fraction of a degree), so to first order

        grid_sampler(x) ≈ add_7(x) + δ(x) · ∇(add_7)(x)
        |Δ(x)|          ≈ |δ(x)| · |∇(add_7)(x)|

    where δ(x) is the per-pixel displacement the warp applies (position-
    dependent: e.g. a uniform scale displaces points near the grid's edges
    more than points near its centre). `grid` -- the STN's *actual* sampling
    grid for THIS image -- lets us reconstruct the real δ(x) (via
    `displacement_field`) rather than assume it's spatially uniform. We then
    compare two predictors of |Δ| against the ground truth, both via Pearson
    correlation across all 900 pixels of each top-N channel:
      - |∇before| alone           (ignores where in the grid x is)
      - |∇before| · |δ(x)|        (the full first-order model)
    If the latter correlates substantially better, that's evidence the warp
    really does behave like a textbook small affine perturbation on these
    channels; if not, something less trivial is going on.
    """
    diff   = np.abs(before.astype(np.float64) - after.astype(np.float64))
    flat   = diff.reshape(diff.shape[0], -1)
    d_mean = flat.mean(axis=1)
    d_max  = flat.max(axis=1)

    order = np.argsort(-d_mean)[:top_n]

    have_grid = grid is not None and grid.shape[:2] == before.shape[1:3]
    disp = displacement_field(grid) if have_grid else np.ones(before.shape[1:3], dtype=np.float64)
    disp_hi = float(disp.max())
    displacement_png = (
        render_tile(disp, colormap, tile_px, 0.0, disp_hi if disp_hi > 1e-9 else 1.0)
        if have_grid else None
    )

    items, grad_corrs, pred_corrs = [], [], []
    for c in order:
        c = int(c)
        b, a, d = before[c], after[c], diff[c]
        lo = float(min(b.min(), a.min()))
        hi = float(max(b.max(), a.max()))
        d_hi = float(d.max())

        # Sobel gradient magnitude of the PRE-warp channel -- ∇(add_7).
        gx = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(b.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        # Full first-order prediction: |∇before| · |δ(x)| (or just |∇before|
        # if no grid was available -- then pred_corr === grad_corr).
        pred = grad * disp
        p_hi = float(pred.max())

        flat_d, flat_g, flat_p = d.flatten(), grad.flatten(), pred.flatten()
        grad_corr = _safe_corr(flat_d, flat_g)
        pred_corr = _safe_corr(flat_d, flat_p)
        if grad_corr is not None: grad_corrs.append(grad_corr)
        if pred_corr is not None: pred_corrs.append(pred_corr)

        items.append({
            "idx":         c,
            "diff_mean":   round(float(d_mean[c]), 4),
            "diff_max":    round(float(d_max[c]), 4),
            "before_png":  render_tile(b, colormap, tile_px, lo, hi),
            "after_png":   render_tile(a, colormap, tile_px, lo, hi),
            "diff_png":    render_tile(d, colormap, tile_px, 0.0, d_hi if d_hi > 1e-9 else 1.0),
            "pred_png":    render_tile(pred, colormap, tile_px, 0.0, p_hi if p_hi > 1e-9 else 1.0),
            "shared_min":  round(lo, 4),
            "shared_max":  round(hi, 4),
            "grad_corr":   round(grad_corr, 3) if grad_corr is not None else None,
            "pred_corr":   round(pred_corr, 3) if pred_corr is not None else None,
        })

    return {
        "items":             items,
        "displacement_png":  displacement_png,
        "used_displacement": have_grid,
        "mean_grad_corr":    round(float(np.mean(grad_corrs)), 3) if grad_corrs else None,
        "mean_pred_corr":    round(float(np.mean(pred_corrs)), 3) if pred_corrs else None,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        families=list(MODELS.keys()),
        precisions=["fp32", "int8"],
        layers_by_family=LAYERS_BY_FAMILY,
        colormaps=list(COLORMAPS.keys()),
    )


@app.route("/api/feature_maps", methods=["POST"])
def feature_maps():
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded"}), 400

    family     = request.form.get("family",     "stn-fomo")
    precision  = request.form.get("precision",  "fp32")
    layer      = request.form.get("layer",      next(iter(LAYERS_BY_FAMILY["stn-fomo"])))
    colormap   = request.form.get("colormap",   "viridis")
    norm_mode  = request.form.get("norm",       "per_channel")
    tile_px    = int(request.form.get("tile",   84))
    bypass_stn = request.form.get("bypass_stn", "0") == "1"

    # Bypass is only meaningful for families that have a GridSample (STN models).
    can_bypass = family in ROTATION_DIFF_GRID_LAYER
    bypass_stn = bypass_stn and can_bypass

    if family not in MODELS:
        return jsonify({"error": f"unknown family {family}"}), 400
    if precision not in MODELS[family]:
        return jsonify({"error": f"unknown precision {precision}"}), 400
    if layer not in LAYERS_BY_FAMILY[family]:
        return jsonify({"error": f"layer {layer} not valid for {family}"}), 400

    data = request.files["image"].read()
    arr  = np.frombuffer(data, np.uint8)
    bgr  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "could not decode image"}), 400

    blob, preview_rgb = preprocess(bgr)

    sess      = get_session(family, precision, layer, bypass=bypass_stn)
    in_name   = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    results   = sess.run(out_names, {in_name: blob})
    named     = dict(zip(out_names, results))

    info      = MODEL_INFO[family]
    num_maps  = info["num_maps"]
    grid_size = info["grid"]   # spatial grid dimension (30 or 60)
    bg_idx    = info["bg_idx"]

    fmap = np.asarray(named[layer])[0]  # (C, H, W)
    if fmap.shape[0] != num_maps:
        return jsonify(
            {"error": f"layer {layer} has {fmap.shape[0]} channels, expected {num_maps}"}
        ), 500

    # Sanity-check: in bypass mode the grid_sampler output should be nearly
    # identical to add_7 (identity grid_sample ≈ passthrough).  No action
    # needed here; the rotation-diff section will naturally show |Δ| ≈ 0.

    g_min = float(fmap.min())
    g_max = float(fmap.max())

    tiles = []
    for c in range(num_maps):
        ch    = fmap[c]
        c_min = float(ch.min())
        c_max = float(ch.max())
        vmin, vmax = (g_min, g_max) if norm_mode == "global" else (c_min, c_max)
        tiles.append({
            "idx":    c,
            "png":    render_tile(ch, colormap, tile_px, vmin, vmax),
            "min":    round(c_min, 4),
            "max":    round(c_max, 4),
            "mean":   round(float(ch.mean()), 4),
            "active": bool(c_max > 1e-6),
        })

    # Detection output -- handle background channel per family.
    det = np.asarray(named["output"])[0]
    det = sigmoid(det) if (det.min() < 0.0 or det.max() > 1.0) else det

    # For FOMO the output is [bg, cls0, cls1, cls2, cls3]; skip bg for display.
    # For STN-FOMO the output is [cls0, cls1, cls2, cls3]; no skip needed.
    cls_offset = 1 if bg_idx is not None else 0
    heat_px    = max(tile_px, 110)

    class_scores = [
        {"name": CLASSES[i], "max": round(float(det[i + cls_offset].max()), 4)}
        for i in range(len(CLASSES))
    ]
    class_heatmaps = [
        {
            "name": CLASSES[i],
            "max":  round(float(det[i + cls_offset].max()), 4),
            "mean": round(float(det[i + cls_offset].mean()), 4),
            "png":  render_tile(det[i + cls_offset], colormap, heat_px, 0.0, 1.0),
        }
        for i in range(len(CLASSES))
    ]

    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)

    # "Most affected by the warp": rank backbone channels by how much the STN
    # changed them (add_7 before vs grid_sampler after), using the SAME
    # forward pass -- only possible when the family exposes both layers and
    # they came back with matching shapes.
    rotation_diff = None
    diff_layers = ROTATION_DIFF_LAYERS.get(family)
    if diff_layers and all(name in named for name in diff_layers):
        before_name, after_name = diff_layers
        before = np.asarray(named[before_name])[0]
        after  = np.asarray(named[after_name])[0]
        if before.shape == after.shape:
            # Pull the STN's ACTUAL per-image sampling grid (if exposed for
            # this family) so the displacement field used in the first-order
            # prediction reflects the real, learned, input-dependent warp --
            # not an assumed/average one.
            grid_layer_name = ROTATION_DIFF_GRID_LAYER.get(family)
            grid = None
            if grid_layer_name and grid_layer_name in named:
                g = np.asarray(named[grid_layer_name])
                if g.ndim == 4 and g.shape[0] == 1 and g.shape[-1] == 2:
                    grid = g[0]

            rd = build_rotation_diff(before, after, colormap, heat_px, grid=grid)
            rotation_diff = {
                "before_layer": before_name,
                "after_layer":  after_name,
                "grid_layer":   grid_layer_name if rd["used_displacement"] else None,
                **rd,
            }

    return jsonify({
        "family":        family,
        "precision":     precision,
        "layer":         layer,
        "layer_label":   LAYERS_BY_FAMILY[family][layer],
        "grid":          grid_size,
        "bypass_stn":    bool(bypass_stn),
        "num_maps":      num_maps,
        "global_min":    round(g_min, 4),
        "global_max":    round(g_max, 4),
        "input_preview": png_b64(preview_bgr),
        "class_scores":  class_scores,
        "class_heatmaps":class_heatmaps,
        "tiles":         tiles,
        "rotation_diff": rotation_diff,
    })


if __name__ == "__main__":
    print("FOMO / STN-FOMO feature-map inspector")
    for fam, variants in MODELS.items():
        for prec, p in variants.items():
            print(f"  {fam:10s} {prec:5s} {'OK ' if os.path.exists(p) else 'MISSING'} {p}")
    app.run(host="127.0.0.1", port=5005, debug=False, threaded=True)
