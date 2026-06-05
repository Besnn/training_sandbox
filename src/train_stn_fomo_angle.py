"""Train + evaluate STN-FOMO at 480x480 with an ANGLE output SUPERVISED from OBB.

Angle-aware STN-FOMO whose forward returns (class_logits, angle). The STN's
predicted rotation is now TRAINED against the railroad-crossing's oriented
bounding box: the dataset's OBB-polygon labels give a ground-truth orientation
theta*, and an angle loss pulls the model's predicted angle toward it.

COUPLED cos/sin:
    The localizer regresses a raw (cos, sin) that is L2-normalized onto the unit
    circle (cos^2 + sin^2 == 1), so the warp is a pure rotation and
    angle = atan2(sin, cos) is always a valid, non-bogus angle. The same coupled
    pair drives both the warp and the reported/supervised angle.

ANGLE SUPERVISION (pi-symmetric double-angle loss):
    theta* is the orientation of the LONG edge of the railroad-crossing OBB
    polygon, measured in the same square-resized image space the model sees (it
    is computed from the NORMALIZED corner coords, so the resize-to-square cancels).
    A rectangle has 180-degree symmetry (theta and theta+pi are the same box), so
    we use a wrap-around-safe DOUBLE-ANGLE loss:
        L_angle = 1 - cos(2 * (theta_pred - theta*))
    which is 0 when the orientations match (incl. the 180-degree flip) and 2 when
    they are perpendicular. Images without a railroad-crossing OBB are masked out.
    Total loss = focal(centroids) + ANGLE_LOSS_WEIGHT * L_angle.

    Set ANGLE_LOSS_WEIGHT = 0.0 to recover the original self-supervised behaviour.
    ANGLE_SOURCE_CLASS selects which class' box provides the orientation.

Apple Silicon note:
    grid_sample's backward (and atan2) have no MPS kernel; PYTORCH_ENABLE_MPS_FALLBACK=1
    (set before importing torch) routes only those ops to CPU; the backbone stays on MPS.
"""

import os

# Must be set BEFORE torch is imported so the MPS->CPU fallback registers.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import glob
import math

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

# Per-class Focal Loss balancing (centroid head).
CROSSING_CLASS_NAME = "railroad-crossing"
CROSSING_ALPHA = 0.75
STATIC_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# Angle supervision from the OBB polygon orientation.
ANGLE_SOURCE_CLASS = "railroad-crossing"   # whose OBB long edge defines theta*
ANGLE_LOSS_WEIGHT = 1.0                     # 0.0 -> self-supervised (no angle loss)

# Per-cell decision threshold for evaluation (sigmoid space).
CONF_THRESHOLD = 0.3


# --- OBB ORIENTATION HELPER ------------------------------------------------
def polygon_long_edge_angle(coords):
    """Orientation (radians) of the LONGER side of a 4-corner OBB polygon.

    coords: [x1,y1,x2,y2,x3,y3,x4,y4] in NORMALIZED image coords. Returns the
    angle of the longer of the two adjacent edges (p1->p2 vs p2->p3), or None if
    the box is degenerate. Normalized coords make this the orientation in the
    square-resized image the model sees (the 480x scaling cancels in atan2).
    """
    pts = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
    e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])   # p1 -> p2
    e2 = (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])   # p2 -> p3
    l1 = e1[0] * e1[0] + e1[1] * e1[1]
    l2 = e2[0] * e2[0] + e2[1] * e2[1]
    if max(l1, l2) < 1e-10:
        return None
    edge = e1 if l1 >= l2 else e2
    return math.atan2(edge[1], edge[0])


# --- ARCHITECTURE: STN + FOMO 1/16 HEAD, WITH ANGLE OUTPUT -----------------
class STN_FOMO_Angle_480(nn.Module):
    """STN-FOMO whose forward returns (class_logits, angle). angle is the STN's
    predicted scene rotation in RADIANS, from a unit-circle-coupled (cos, sin)."""

    def __init__(self, num_classes, head_channels=96):
        super().__init__()
        backbone = models.mobilenet_v2(weights="DEFAULT").features
        self.features = nn.Sequential(*list(backbone.children())[:14])
        backbone_channels = 96

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.loc_fc = nn.Sequential(
            nn.Linear(backbone_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        self._init_identity_transform()

        self.head = nn.Sequential(
            nn.Conv2d(backbone_channels, backbone_channels, kernel_size=3,
                      padding=1, groups=backbone_channels),   # depthwise
            nn.ReLU(inplace=True),
            nn.Conv2d(backbone_channels, head_channels, kernel_size=1),  # pointwise mix
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, num_classes, kernel_size=1),        # classifier
        )

    def _init_identity_transform(self):
        # Small-random output weights (NOT zeros): zero weights make the predicted
        # angle CONSTANT (== bias) regardless of input, which both kills input
        # dependence and, combined with a 0-deg bias, can park training on the
        # double-angle loss plateau (90 deg off => ~zero gradient). Small weights
        # keep the warp near-identity while letting the angle vary per image.
        last_fc = self.loc_fc[-1]
        nn.init.normal_(last_fc.weight, std=0.01)
        with torch.no_grad():
            last_fc.bias.copy_(torch.tensor([1.0, 0.0]))   # cos=1, sin=0 -> angle 0

    def init_angle_bias(self, theta0):
        """Set the localizer bias so the mean predicted angle starts at theta0
        (radians). Used to seed supervision at the dataset's mean OBB orientation
        so the double-angle loss starts in its basin, not on the 90-deg plateau."""
        with torch.no_grad():
            self.loc_fc[-1].bias.copy_(
                torch.tensor([math.cos(theta0), math.sin(theta0)], dtype=torch.float32)
            )

    def _predict_rotation(self, feat, eps=1e-6):
        """Regress + COUPLE the rotation. Returns (cos, sin, angle), each (B, 1).
        cos/sin are L2-normalized so cos^2+sin^2==1; angle = atan2(sin, cos)."""
        descriptor = self.global_pool(feat).flatten(1)
        raw = self.loc_fc(descriptor)
        raw_cos, raw_sin = raw[:, 0:1], raw[:, 1:2]
        norm = torch.sqrt(raw_cos * raw_cos + raw_sin * raw_sin + eps)
        cos = raw_cos / norm
        sin = raw_sin / norm
        angle = torch.atan2(sin, cos)
        return cos, sin, angle

    def _rotation_grid(self, cos, sin, height, width, device, dtype):
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.unsqueeze(0)
        grid_y = grid_y.unsqueeze(0)
        cos = cos.view(-1, 1, 1)
        sin = sin.view(-1, 1, 1)
        rot_x = cos * grid_x - sin * grid_y
        rot_y = sin * grid_x + cos * grid_y
        return torch.stack((rot_x, rot_y), dim=-1)

    def forward(self, x):
        feat = self.features(x)
        _, _, h, w = feat.shape
        cos, sin, angle = self._predict_rotation(feat)
        grid = self._rotation_grid(cos, sin, h, w, feat.device, feat.dtype)
        rectified = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        logits = self.head(rectified)
        return logits, angle                                     # angle: (B, 1) radians


# --- DATASET: HARD-CENTROID GRID + OBB ANGLE TARGET ------------------------
class YOLOCentroidAngleDataset(Dataset):
    """Yields (image, target_grid, gt_angle, has_angle).

    target_grid: (num_classes, grid, grid) hard binary centroids (one 1.0 cell per
    object). gt_angle: (1,) radians, the long-edge orientation of the first
    `angle_source_id` OBB polygon in the image. has_angle: (1,) {0.,1.} mask — 0
    when the image has no such polygon (then the angle loss ignores it).
    """

    def __init__(self, split_root, num_classes, img_size=IMG_SIZE, grid_size=GRID_SIZE,
                 angle_source_id=0):
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
        self.angle_source_id = angle_source_id
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        if not self.img_paths:
            print(f"\n[!] ERROR: no images found in {self.img_dir}")
            sys.exit(1)

    def __len__(self):
        return len(self.img_paths)

    def _parse_label(self, label_path):
        """Return (centroids, gt_angle_or_None).

        centroids: [(cls_id, x_c, y_c), ...]. gt_angle: long-edge angle of the
        FIRST `angle_source_id` polygon (needs the 9-field OBB form), else None.
        """
        centroids = []
        gt_angle = None
        if not os.path.exists(label_path):
            return centroids, gt_angle
        with open(label_path, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                cls_id = int(parts[0])
                if len(parts) == 9:                       # OBB polygon
                    coords = [float(v) for v in parts[1:9]]
                    x_c = sum(coords[i] for i in (0, 2, 4, 6)) / 4.0
                    y_c = sum(coords[i] for i in (1, 3, 5, 7)) / 4.0
                    if cls_id == self.angle_source_id and gt_angle is None:
                        gt_angle = polygon_long_edge_angle(coords)
                else:                                     # centroid: cls x_c y_c [w h]
                    x_c, y_c = float(parts[1]), float(parts[2])
                centroids.append((cls_id, x_c, y_c))
        return centroids, gt_angle

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        label_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        centroids, gt_angle = self._parse_label(os.path.join(self.label_dir, label_name))

        gs = self.grid_size
        target = torch.zeros((self.num_classes, gs, gs), dtype=torch.float32)
        for cls_id, x_c, y_c in centroids:
            if not (0 <= cls_id < self.num_classes):
                continue
            gx = min(int(x_c * gs), gs - 1)
            gy = min(int(y_c * gs), gs - 1)
            target[cls_id, gy, gx] = 1.0

        has_angle = 1.0 if gt_angle is not None else 0.0
        angle_t = torch.tensor([gt_angle if gt_angle is not None else 0.0], dtype=torch.float32)
        return self.transform(img), target, angle_t, torch.tensor([has_angle], dtype=torch.float32)


# --- LOSSES ----------------------------------------------------------------
def per_cell_focal_loss(logits, target, alpha, gamma=FOCAL_GAMMA):
    """Per-cell focal loss; alpha is the (C,) per-class positive-weight vector."""
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha = alpha.view(1, -1, 1, 1)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    num_pos = target.eq(1.0).sum().clamp(min=1.0)
    return loss.sum() / num_pos


def angle_double_cos_loss(pred_angle, gt_angle, has_angle):
    """pi-symmetric orientation loss: mean over valid samples of 1 - cos(2*(pred-gt)).

    The double angle makes theta and theta+pi equivalent (rectangle symmetry) and
    is wrap-around safe. `has_angle` (B,1) masks images with no source OBB.
    """
    delta = 2.0 * (pred_angle - gt_angle)
    per_sample = 1.0 - torch.cos(delta)          # (B,1) in [0, 2]
    denom = has_angle.sum().clamp(min=1.0)
    return (per_sample * has_angle).sum() / denom


def orientation_error_deg(pred_angle, gt_angle):
    """Smallest angle between two orientations (pi-symmetric), degrees in [0, 90]."""
    delta = pred_angle - gt_angle
    wrapped = 0.5 * torch.atan2(torch.sin(2.0 * delta), torch.cos(2.0 * delta))
    return torch.abs(wrapped) * (180.0 / math.pi)


def build_class_alpha(class_names):
    alpha = [CROSSING_ALPHA if name == CROSSING_CLASS_NAME else STATIC_ALPHA
             for name in class_names]
    return torch.tensor(alpha, dtype=torch.float32)


def estimate_mean_orientation(dataset, max_samples=400):
    """pi-symmetric circular mean of the dataset's GT orientations (radians).

    Averages (cos 2theta, sin 2theta) over images that have a source-class OBB,
    then halves the resulting angle. Used to seed the localizer bias so angle
    supervision starts in the loss basin. Returns 0.0 if no GT angles exist.
    """
    c2 = s2 = 0.0
    n = 0
    step = max(1, len(dataset) // max_samples)
    for i in range(0, len(dataset), step):
        _, _, ang, has = dataset[i]
        if has.item() > 0.5:
            th = float(ang)
            c2 += math.cos(2.0 * th)
            s2 += math.sin(2.0 * th)
            n += 1
    if n == 0:
        return 0.0
    return 0.5 * math.atan2(s2, c2)


# --- EVALUATION ------------------------------------------------------------
@torch.no_grad()
def build_confusion_matrix(model, loader, num_classes, device, threshold=CONF_THRESHOLD):
    model.eval()
    bg = num_classes
    cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    for imgs, targets, _gt_ang, _has in loader:
        logits, _angle = model(imgs.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
        gts = targets.numpy()
        for prob, gt in zip(probs, gts):
            max_conf = prob.max(axis=0)
            pred_cls = prob.argmax(axis=0)
            pred_label = np.where(max_conf >= threshold, pred_cls, bg)
            gt_active = gt.max(axis=0)
            gt_cls = gt.argmax(axis=0)
            gt_label = np.where(gt_active >= 0.5, gt_cls, bg)
            np.add.at(cm, (gt_label.ravel(), pred_label.ravel()), 1)
    return cm


@torch.no_grad()
def evaluate_angle(model, loader, device):
    """Mean orientation error (deg) vs the OBB GT, over images that have a GT angle."""
    model.eval()
    errs = []
    for imgs, _targets, gt_ang, has in loader:
        _, pred = model(imgs.to(device))
        err = orientation_error_deg(pred.cpu(), gt_ang)          # (B,1)
        mask = has.squeeze(1).bool()
        errs.append(err.squeeze(1)[mask].numpy())
    errs = np.concatenate(errs) if errs else np.array([])
    return errs


def report_angle_eval(errs_deg):
    print("\n--- STN angle vs OBB ground truth (pi-symmetric orientation error) ---")
    if errs_deg.size == 0:
        print("  (no images with a source-class OBB in the val set)")
        return
    print(f"  images with GT angle: {errs_deg.size}")
    print(f"  mean error: {errs_deg.mean():.2f} deg   median: {np.median(errs_deg):.2f} deg   "
          f"90th pct: {np.percentile(errs_deg, 90):.2f} deg")


def report_confusion_matrix(cm, class_names, save_path=None):
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
        print(f"  {name:<22} recall={norm[i, i]:.3f}  (GT cells={int(cm[i].sum())})")

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
    ax.set_title("STN-FOMO (angle) per-cell confusion (row-normalized)")
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
    angle_source_id = class_names.index(ANGLE_SOURCE_CLASS) if ANGLE_SOURCE_CLASS in class_names else 0
    train_path = os.path.join(dataset_root, os.path.basename(data_cfg["train"]))
    val_path = os.path.join(dataset_root, os.path.basename(data_cfg["val"]))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}  (classes={num_classes}: {class_names})")
    print(f"Angle supervised from '{ANGLE_SOURCE_CLASS}' (class {angle_source_id}), "
          f"weight={ANGLE_LOSS_WEIGHT}")
    if device.type == "mps":
        print("Note: grid_sample backward (and atan2) fall back to CPU via "
              "PYTORCH_ENABLE_MPS_FALLBACK; the backbone stays on MPS.")

    model = STN_FOMO_Angle_480(num_classes=num_classes).to(device)

    with torch.no_grad():
        probe_logits, probe_angle = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))
    grid_h, grid_w = probe_logits.shape[-2:]
    if (grid_h, grid_w) != (GRID_SIZE, GRID_SIZE):
        raise RuntimeError(
            f"Backbone produced a {grid_h}x{grid_w} grid, expected "
            f"{GRID_SIZE}x{GRID_SIZE}. Adjust GRID_SIZE or the backbone cut point."
        )
    print(f"Init angle (identity expected ~0 deg): {math.degrees(float(probe_angle)):+.2f} deg")

    train_ds = YOLOCentroidAngleDataset(train_path, num_classes, IMG_SIZE, GRID_SIZE, angle_source_id)
    val_ds = YOLOCentroidAngleDataset(val_path, num_classes, IMG_SIZE, GRID_SIZE, angle_source_id)
    print(f"Images: train={len(train_ds)}, val={len(val_ds)}")

    # Seed the localizer bias at the data's mean OBB orientation so the
    # double-angle loss starts in its basin (not on the 90-deg plateau).
    if ANGLE_LOSS_WEIGHT > 0.0:
        theta0 = estimate_mean_orientation(train_ds)
        model.init_angle_bias(theta0)
        with torch.no_grad():
            seeded = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))[1]
        print(f"Seeded localizer at mean OBB orientation ~{math.degrees(theta0):+.1f} deg "
              f"(model now starts at {math.degrees(float(seeded)):+.1f} deg)")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_alpha = build_class_alpha(class_names).to(device)
    print(f"Focal alpha per class: {dict(zip(class_names, class_alpha.tolist()))}")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        run_focal = run_angle = 0.0
        print(f"\n---> Epoch {epoch + 1}/{EPOCHS}...")

        for i, (imgs, targets, gt_ang, has_ang) in enumerate(train_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            gt_ang, has_ang = gt_ang.to(device), has_ang.to(device)
            optimizer.zero_grad()

            logits, pred_ang = model(imgs)
            if logits.shape[-2:] != targets.shape[-2:]:
                raise RuntimeError(
                    f"Grid mismatch: output {tuple(logits.shape[-2:])} vs target "
                    f"{tuple(targets.shape[-2:])}."
                )

            focal = per_cell_focal_loss(logits, targets, class_alpha)
            angle = angle_double_cos_loss(pred_ang, gt_ang, has_ang)
            loss = focal + ANGLE_LOSS_WEIGHT * angle
            loss.backward()
            optimizer.step()

            run_focal += focal.item()
            run_angle += angle.item()
            if (i + 1) % 2 == 0:
                print(f"      Batch {i + 1}/{len(train_loader)} - "
                      f"focal: {focal.item():.4f}  angle: {angle.item():.4f}")

        n = len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} done - mean focal: {run_focal / n:.4f}  "
              f"mean angle: {run_angle / n:.4f}")

    model_path = os.path.join(src_dir, "stn_fomo_angle_mac.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining complete. Model saved to {model_path}")

    # --- Final evaluation block ---
    print("\n" + "=" * 60)
    print("EVALUATION (validation set)")
    print("=" * 60)
    cm = build_confusion_matrix(model, val_loader, num_classes, device)
    report_confusion_matrix(
        cm, class_names, save_path=os.path.join(src_dir, "stn_fomo_angle_confusion.png")
    )
    report_angle_eval(evaluate_angle(model, val_loader, device))


if __name__ == "__main__":
    train()
