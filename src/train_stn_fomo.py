"""Train + evaluate a rotation-immune STN-FOMO centroid detector at 480x480.

Target hardware (MCU / CPU INT8) has NO Deformable Convolution support, so this
head is built strictly from quantization- and compiler-friendly primitives:
    Conv2d (incl. grouped / depthwise), Linear, (Global) AvgPool2d, bilinear
    grid sampling, and element-wise Add / Sub / Mul / Square / Clamp(ReLU) / Sigmoid.

Instead of deforming the convolution sampling grid (DCNv2), we rectify the WHOLE
feature map with a Spatial Transformer Network (STN): a tiny localization net
predicts the scene orientation, a grid generator builds an inverse-rotation
sampling grid from plain coordinate meshgrids (Mul/Add/Sub only), and a bilinear
sampler warps the features back to an upright 0deg frame. A plain depthwise-
separable head then classifies the upright map. This keeps the long, rotating
'railroad-crossing' learnable while the static classes ('lights-on', 'lights-off',
'trefolo') stay pixel-sharp.

We use a RIGID hard-centroid target grid (one 1.0 cell per object, no smearing).

Apple Silicon note:
    F.grid_sample runs its FORWARD on MPS, but `aten::grid_sampler_2d_backward`
    has no MPS kernel in current builds. We set PYTORCH_ENABLE_MPS_FALLBACK=1
    (before importing torch) so only that backward transparently falls back to
    CPU while the MobileNetV2 backbone keeps running on the GPU. Eval is forward
    only and runs fully on MPS.
"""

import os

# Must be set BEFORE torch is imported so the MPS->CPU fallback registers; it only
# triggers for ops lacking an MPS kernel (here: the grid_sample backward).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import glob

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# --- HYPERPARAMETERS --------------------------------------------------------
IMG_SIZE = 480
GRID_SIZE = 30           # 480 / 16 (backbone cut at features[:14]); verified at runtime
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Per-class Focal Loss balancing. alpha weights the POSITIVE term, so a higher
# alpha pushes the model to fire (suppresses false negatives / background misses).
# 'railroad-crossing' is large, rotating and spatially sparse vs. the 30x30 grid,
# so it gets a high alpha; the static light states keep the standard RetinaNet 0.25.
CROSSING_CLASS_NAME = "railroad-crossing"
CROSSING_ALPHA = 0.75
STATIC_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Per-cell decision threshold for evaluation (sigmoid space).
CONF_THRESHOLD = 0.3


# --- ARCHITECTURE: STN + FOMO 1/16 HEAD ------------------------------------
class STN_FOMO_480(nn.Module):
    """MobileNetV2 (alpha=1.0) backbone cut to 1/16 + an STN rectifier + FOMO head.

    All ops are MCU/INT8-friendly: convs, a small FC localization net, global
    average pooling, bilinear grid sampling, and element-wise math. No DCN.
    """

    def __init__(self, num_classes, head_channels=96):
        super().__init__()

        # MobileNetV2 default weights = alpha 1.0. Cutting at features[:14] gives a
        # 1/16 spatial reduction (480 -> 30x30) with 96 output channels.
        backbone = models.mobilenet_v2(weights="DEFAULT").features
        self.features = nn.Sequential(*list(backbone.children())[:14])
        backbone_channels = 96

        # --- STN localization network: GAP -> FC -> (cos, sin) ---
        # Global Average Pooling collapses the 30x30 map to a 96-vector scene
        # descriptor; the FC head regresses the scene orientation as a (cos, sin)
        # pair. We predict cos/sin directly (rather than an angle) so the deployed
        # graph needs no trigonometric op. We deliberately do NOT renormalize to
        # the unit circle: sqrt/div are not in the allowed primitive set, so the
        # localizer learns a similarity (rotation + mild scale) transform.
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.loc_fc = nn.Sequential(
            nn.Linear(backbone_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        self._init_identity_transform()

        # --- Classification head: depthwise-separable conv -> per-class logits ---
        # Depthwise 3x3 (one filter per channel) + pointwise 1x1 minimizes CPU/MCU
        # cost vs. a dense conv. Returns raw logits; Sigmoid is applied in the loss
        # and at decode. No background channel (background is implicit when low).
        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3,
                      padding=1, groups=backbone_channels),   # depthwise
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, head_channels, kernel_size=1),  # pointwise mix
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, kernel_size=1),        # classifier
        )

    def _init_identity_transform(self):
        """Init the localizer so it starts at the identity (cos=1, sin=0): zero the
        last FC weights and bias it to (1, 0). The STN then begins as a no-op warp
        and learns to rotate from there — the standard stable STN initialization."""
        last_fc = self.loc_fc[-1]
        nn.init.zeros_(last_fc.weight)
        with torch.no_grad():
            last_fc.bias.copy_(torch.tensor([1.0, 0.0]))

    def _rotation_grid(self, cos, sin, height, width, device, dtype):
        """Build a (B, H, W, 2) bilinear sampling grid from primitive meshgrid math.

        Standard normalized [-1, 1] coordinates are inverse-rotated by the predicted
        (cos, sin) using only Multiply / Subtract / Add, so the sampler pulls the
        scene back to an upright orientation:
            x' = cos*x - sin*y
            y' = sin*x + cos*y
        """
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")   # (H, W) each
        grid_x = grid_x.unsqueeze(0)                             # (1, H, W)
        grid_y = grid_y.unsqueeze(0)

        cos = cos.view(-1, 1, 1)                                 # (B, 1, 1)
        sin = sin.view(-1, 1, 1)
        rot_x = cos * grid_x - sin * grid_y                      # (B, H, W)
        rot_y = sin * grid_x + cos * grid_y
        return torch.stack((rot_x, rot_y), dim=-1)              # (B, H, W, 2)

    def forward(self, x):
        feat = self.features(x)                                  # (B, 96, 30, 30)
        _, _, h, w = feat.shape

        # STN: regress orientation, then warp the feature map upright.
        descriptor = self.global_pool(feat).flatten(1)           # (B, 96)
        loc = self.loc_fc(descriptor)                            # (B, 2)
        cos, sin = loc[:, 0:1], loc[:, 1:2]                      # (B, 1, 1)
        grid = self._rotation_grid(cos, sin, h, w, feat.device, feat.dtype)
        rectified = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )

        return self.head(rectified)                              # raw logits (B, C, 30, 30)


# --- DATASET: RIGID HARD-CENTROID GRID -------------------------------------
class YOLOCentroidDataset(Dataset):
    """Yields (image_tensor, hard_target) where target is a strict binary grid.

    target shape: (num_classes, grid, grid). Each object's centre is mapped to a
    SINGLE cell set to exactly 1.0 — no smearing of any kind. Labels may be in
    centroid form (`cls x_c y_c [w h]`) or OBB-polygon form (`cls x1 y1 ... x4 y4`);
    for polygons the centre is the mean of the 4 corners (taking fields 1,2 would
    read a corner, not the centre).
    """

    def __init__(self, split_root, num_classes, img_size=IMG_SIZE, grid_size=GRID_SIZE):
        self.img_dir = os.path.join(split_root, "images")
        self.label_dir = os.path.join(split_root, "labels")
        if not os.path.exists(self.img_dir):
            self.img_dir = split_root
            self.label_dir = split_root.replace("images", "labels")

        self.img_paths = sorted([
            p for p in glob.glob(os.path.join(self.img_dir, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        self.num_classes = num_classes
        self.grid_size = grid_size

        # ToTensor() already scales pixels 0-255 -> 0.0-1.0; no ImageNet normalize.
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        if not self.img_paths:
            print(f"\n[!] ERROR: no images found in {self.img_dir}")
            sys.exit(1)

    def __len__(self):
        return len(self.img_paths)

    def _read_centroids(self, label_path):
        """Return [(cls_id, x_c, y_c), ...] in normalized [0,1] image coords."""
        out = []
        if not os.path.exists(label_path):
            return out
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                cls_id = int(parts[0])
                if len(parts) == 9:                       # OBB polygon: 4 corners
                    x_c = sum(float(parts[i]) for i in (1, 3, 5, 7)) / 4.0
                    y_c = sum(float(parts[i]) for i in (2, 4, 6, 8)) / 4.0
                else:                                     # centroid: cls x_c y_c [w h]
                    x_c, y_c = float(parts[1]), float(parts[2])
                out.append((cls_id, x_c, y_c))
        return out

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        gs = self.grid_size
        target = torch.zeros((self.num_classes, gs, gs), dtype=torch.float32)
        for cls_id, x_c, y_c in self._read_centroids(label_path):
            if not (0 <= cls_id < self.num_classes):
                continue
            # Hard assignment: one cell, value 1.0. No Gaussian/Manhattan disk.
            gx = min(int(x_c * gs), gs - 1)
            gy = min(int(y_c * gs), gs - 1)
            target[cls_id, gy, gx] = 1.0

        return self.transform(img), target


# --- LOSS: PER-CELL SIGMOID FOCAL LOSS -------------------------------------
def per_cell_focal_loss(logits, target, alpha, gamma=FOCAL_GAMMA):
    """Focal loss treating every grid cell as an independent binary classifier.

    logits, target: (B, C, H, W). alpha: (C,) per-class positive-weight vector on
    the same device. Normalized by the number of positive cells (RetinaNet-style),
    which keeps the loss stable despite the heavy positive/background imbalance of
    a sparse hard-centroid grid.
    """
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    # p_t and alpha_t selected per cell by whether it is a positive or a negative.
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha = alpha.view(1, -1, 1, 1)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)

    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    num_pos = target.eq(1.0).sum().clamp(min=1.0)
    return loss.sum() / num_pos


def build_class_alpha(class_names):
    """Per-class focal alpha: high for the rotating crossing, standard for statics."""
    alpha = [CROSSING_ALPHA if name == CROSSING_CLASS_NAME else STATIC_ALPHA
             for name in class_names]
    return torch.tensor(alpha, dtype=torch.float32)


# --- EVALUATION: ROW-NORMALIZED CONFUSION MATRIX ---------------------------
@torch.no_grad()
def build_confusion_matrix(model, loader, num_classes, device, threshold=CONF_THRESHOLD):
    """Per-cell confusion matrix of shape (num_classes+1, num_classes+1).

    The last row/col is 'background'. For each cell: the predicted label is the
    argmax class when its sigmoid >= threshold, else background; the GT label is
    the cell's hard class, else background. This captures both false negatives
    (class row -> background col) and false positives (background row -> class col).
    """
    model.eval()
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    for imgs, targets in loader:
        probs = torch.sigmoid(model(imgs.to(device))).cpu().numpy()  # (B, C, H, W)
        gts = targets.numpy()
        for prob, gt in zip(probs, gts):
            max_conf = prob.max(axis=0)            # (H, W) strongest class confidence
            pred_cls = prob.argmax(axis=0)         # (H, W) winning class index
            pred_label = np.where(max_conf >= threshold, pred_cls, bg)

            gt_active = gt.max(axis=0)             # (H, W) 1.0 where any class fires
            gt_cls = gt.argmax(axis=0)
            gt_label = np.where(gt_active >= 0.5, gt_cls, bg)

            np.add.at(cm, (gt_label.ravel(), pred_label.ravel()), 1)
    return cm


def report_confusion_matrix(cm, class_names, save_path=None):
    """Print a row-normalized confusion matrix (+ per-class recall) and optionally
    save a heatmap PNG."""
    labels = list(class_names) + ["background"]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(row_sums, 1)

    width = max(max(len(l) for l in labels), 10)
    print("\n--- Per-cell confusion matrix (row-normalized; "
          f"conf>={CONF_THRESHOLD}) ---")
    print(" " * (width + 2) + "".join(f"{l:>{width + 2}}" for l in labels))
    for i, row_label in enumerate(labels):
        cells = "".join(f"{norm[i, j]:>{width + 1}.3f} " for j in range(len(labels)))
        print(f"{row_label:<{width + 2}}{cells}")

    print("\nPer-class recall (diagonal) and support:")
    for i, name in enumerate(class_names):
        support = int(cm[i].sum())
        recall = norm[i, i]
        print(f"  {name:<22} recall={recall:.3f}  (GT cells={support})")

    if save_path is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (matplotlib unavailable, skipping {save_path}: {exc})")
        return

    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.2 * len(labels) + 2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("STN-FOMO per-cell confusion (row-normalized)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{100 * norm[i, j]:.1f}%\n({int(cm[i, j])})",
                    ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix heatmap saved to: {save_path}")


# --- TRAINING ---------------------------------------------------------------
def train():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(src_dir, "datasets", "split_centroid_dataset")
    yaml_path = os.path.join(dataset_root, "data.yaml")

    print("\n" + "=" * 60)
    print(f"SCRIPT DIR : {src_dir}")
    print(f"DATASET    : {dataset_root}")
    print("=" * 60 + "\n")

    if not os.path.exists(yaml_path):
        print(f"ERROR: data.yaml not found at {yaml_path}")
        sys.exit(1)

    with open(yaml_path, "r") as f:
        data_cfg = yaml.safe_load(f)

    class_names = list(data_cfg["names"])
    num_classes = len(class_names)
    train_path = os.path.join(dataset_root, os.path.basename(data_cfg["train"]))
    val_path = os.path.join(dataset_root, os.path.basename(data_cfg["val"]))

    # Native Apple Silicon acceleration; grid_sample backward falls back to CPU.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  (classes={num_classes}: {class_names})")
    if device.type == "mps":
        print("Note: grid_sample backward has no MPS kernel and runs on CPU via "
              "PYTORCH_ENABLE_MPS_FALLBACK; the backbone stays on MPS.")

    model = STN_FOMO_480(num_classes=num_classes).to(device)

    # Verify the backbone really lands on a GRID_SIZE x GRID_SIZE grid for this input.
    with torch.no_grad():
        probe = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
    grid_h, grid_w = probe.shape[-2:]
    if (grid_h, grid_w) != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(
            f"Backbone produced a {grid_h}x{grid_w} grid, expected "
            f"{GRID_SIZE}x{GRID_SIZE}. Adjust GRID_SIZE or the backbone cut point."
        )

    train_ds = YOLOCentroidDataset(train_path, num_classes, IMG_SIZE, GRID_SIZE)
    val_ds = YOLOCentroidDataset(val_path, num_classes, IMG_SIZE, GRID_SIZE)
    print(f"Images: train={len(train_ds)}, val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_alpha = build_class_alpha(class_names).to(device)
    print(f"Focal alpha per class: {dict(zip(class_names, class_alpha.tolist()))}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        print(f"\n---> Epoch {epoch + 1}/{EPOCHS}...")

        for i, (imgs, targets) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()

            logits = model(imgs)
            if logits.shape[-2:] != targets.shape[-2:]:
                raise RuntimeError(
                    f"Grid mismatch: output {tuple(logits.shape[-2:])} vs target "
                    f"{tuple(targets.shape[-2:])}."
                )

            loss = per_cell_focal_loss(logits, targets, class_alpha)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 2 == 0:
                print(f"      Batch {i + 1}/{len(train_loader)} - loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{EPOCHS} done - mean loss: "
              f"{running_loss / len(train_loader):.4f}")

    model_path = os.path.join(src_dir, "stn_fomo_mac.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to {model_path}")

    # --- Final evaluation block ---
    print("\n" + "=" * 60)
    print("EVALUATION (validation set)")
    print("=" * 60)
    cm = build_confusion_matrix(model, val_loader, num_classes, device)
    report_confusion_matrix(
        cm, class_names, save_path=os.path.join(src_dir, "stn_fomo_confusion.png")
    )


if __name__ == "__main__":
    train()
