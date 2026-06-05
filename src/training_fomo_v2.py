import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import glob
import sys

# --- TARGET SMEARING (CenterNet-style soft Gaussian heatmaps) ---
# Instead of marking a single hard cell per centroid, each centroid is splatted
# as a 2D Gaussian (peak 1.0 at the centre cell, decaying outward). The model
# has NO background channel: it predicts one sigmoid heatmap per class and is
# trained with penalty-reduced focal loss. This gives a tolerant target that
# helps the model fire on hard cases (e.g. tilted / large signs) where the exact
# centre cell is ambiguous.
#
# GAUSS_SIGMA is the smear width in GRID CELLS. ~1 cell => a usable bump spanning
# roughly +/-3 cells. Larger = more tolerant but blurrier peaks (risk of merging
# two nearby same-class objects); smaller = sharper but closer to a hard target.
GAUSS_SIGMA = 1.0


def draw_gaussian(heatmap, cx, cy, sigma):
    """Max-merge a 2D Gaussian (peak 1.0 at integer cell cx,cy) into heatmap[H,W]."""
    h, w = heatmap.shape
    radius = max(1, int(round(3 * sigma)))
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys = torch.arange(y0, y1, dtype=torch.float32).unsqueeze(1)
    xs = torch.arange(x0, x1, dtype=torch.float32).unsqueeze(0)
    g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))
    region = heatmap[y0:y1, x0:x1]
    heatmap[y0:y1, x0:x1] = torch.maximum(region, g)


# --- BACKBONE: MobileNetV2 alpha=0.35 from a local ImageNet-pretrained checkpoint ---
# We use alpha=0.35 (not torchvision's default 1.0) and initialise the backbone
# from etc/mobilenetv2_0.35-*.pth (ImageNet-pretrained). That file uses a
# non-torchvision MobileNetV2 layout (flat Conv/BN/ReLU6 modules), so its keys
# must be remapped to torchvision's fused ConvBNReLU layout before loading.
MNV2_035_WEIGHTS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "etc", "mobilenetv2_0.35-b2e15951.pth",
)


def _remap_mobilenet_v2_keys(state_dict):
    """Non-torchvision MobileNetV2 state_dict -> torchvision ConvBNReLU key layout.

    Only the feature extractor is kept; the classifier and the final 1x1 conv
    (features.18 / `conv.*`) are dropped because FOMO cuts the backbone before
    them. ReLU6 layers carry no parameters and are skipped.
    """
    remapped = {}
    for key, value in state_dict.items():
        if not key.startswith("features.") or key.startswith("features.18"):
            continue
        parts = key.split(".")
        block = parts[1]
        if block == "0" or parts[2] != "conv":   # stem ConvBNReLU already matches
            remapped[key] = value
            continue
        sub_idx, rest = parts[3], ".".join(parts[4:])
        has_expand = any(k.startswith(f"features.{block}.conv.6.") for k in state_dict)
        index_map = ({"0": "0.0", "1": "0.1", "3": "1.0", "4": "1.1", "6": "2", "7": "3"}
                     if has_expand else
                     {"0": "0.0", "1": "0.1", "3": "1", "4": "2"})
        if sub_idx not in index_map:              # ReLU6 -> no params
            continue
        new_key = f"features.{block}.conv.{index_map[sub_idx]}"
        remapped[new_key + ("." + rest if rest else "")] = value
    return remapped


def load_mobilenet_v2_035(weights_path=MNV2_035_WEIGHTS_PATH):
    """Return MobileNetV2 alpha=0.35 `.features`, initialised from the local checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"MobileNetV2 alpha=0.35 weights not found at {weights_path}. "
            f"Place mobilenetv2_0.35-*.pth there (see etc/)."
        )
    net = models.mobilenet_v2(weights=None, width_mult=0.35)
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, _ = net.load_state_dict(_remap_mobilenet_v2_keys(state_dict), strict=False)
    # Everything except the unused final conv (features.18) must be populated.
    leftover = [k for k in missing
                if k.startswith("features.") and not k.startswith("features.18")]
    if leftover:
        raise RuntimeError(f"Checkpoint did not populate backbone layers: {leftover[:5]}")
    return net.features


class FOMO_PL_480(nn.Module):
    def __init__(self, num_classes, weights_path=MNV2_035_WEIGHTS_PATH,
                 width_mult=0.35, pretrained=True,
                 cut_index=14, head_mid=96, head_out=None):
        super(FOMO_PL_480, self).__init__()
        # Backbone. Training (the default) uses MobileNetV2 alpha=0.35 initialised
        # from the local ImageNet-pretrained checkpoint. To rebuild a model for
        # conversion / eval of an EXISTING checkpoint, pass pretrained=False (the
        # trained weights overwrite the backbone anyway, so no pretrained file is
        # needed) and the width_mult that matches that checkpoint (1.0 for models
        # trained before the alpha=0.35 switch).
        if not pretrained:
            backbone = models.mobilenet_v2(weights=None, width_mult=width_mult).features
        elif abs(width_mult - 0.35) < 1e-6:
            backbone = load_mobilenet_v2_035(weights_path)
        else:
            tv_weights = "DEFAULT" if abs(width_mult - 1.0) < 1e-6 else None
            backbone = models.mobilenet_v2(weights=tv_weights, width_mult=width_mult).features

        # Cut point sets the output stride / grid resolution:
        #   cut_index=14 -> 1/16 (480 -> 30x30, current head)
        #   cut_index=7  -> 1/8  (480 -> 60x60, older shallower head)
        # The default trains the 1/16 model; from_checkpoint() restores whatever
        # cut a saved checkpoint actually used so any variant rebuilds correctly.
        self.features = nn.Sequential(*list(backbone.children())[:cut_index])

        # Infer the cut's output channels instead of hardcoding, so the head
        # matches whatever backbone+cut was built.
        backbone_channels = self._infer_backbone_channels()

        # La testa esatta di FOMO: convoluzioni 1x1, no dropout. head_out defaults
        # to num_classes (CenterNet heatmap, no background channel); an older
        # softmax checkpoint uses num_classes+1 and is detected by from_checkpoint().
        # The model returns raw logits; sigmoid is applied in the loss and at decode.
        out_channels = num_classes if head_out is None else head_out
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=backbone_channels, out_channels=head_mid, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=head_mid, out_channels=out_channels, kernel_size=1, stride=1)
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path, num_classes, map_location="cpu"):
        """Rebuild a model matching a saved checkpoint and load it, regardless of
        backbone width, cut depth, or head shape. Everything is read back from the
        checkpoint tensor shapes (the checkpoint supplies every weight, so no
        pretrained backbone file/download is needed):
          - width:    stem conv channels (16 -> alpha 0.35, 32 -> alpha 1.0)
          - cut depth: highest features.N block index present (7 -> 1/8, 14 -> 1/16)
          - head:     head.0 / head.2 conv shapes (mid channels, and out =
                      num_classes for the heatmap head or num_classes+1 for the
                      older softmax head)."""
        state_dict = torch.load(checkpoint_path, map_location=map_location)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        stem_channels = state_dict["features.0.0.weight"].shape[0]
        width_by_stem = {16: 0.35, 32: 1.0}
        if stem_channels not in width_by_stem:
            raise ValueError(
                f"Cannot infer MobileNetV2 width from stem channels={stem_channels} "
                f"in {checkpoint_path}."
            )

        feature_blocks = [int(k.split(".")[1]) for k in state_dict
                          if k.startswith("features.")]
        cut_index = max(feature_blocks) + 1
        head_mid = state_dict["head.0.weight"].shape[0]
        head_out = state_dict["head.2.weight"].shape[0]

        model = cls(num_classes=num_classes,
                    width_mult=width_by_stem[stem_channels], pretrained=False,
                    cut_index=cut_index, head_mid=head_mid, head_out=head_out)
        model.load_state_dict(state_dict)
        return model

    def _infer_backbone_channels(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            channels = self.features(torch.zeros(1, 3, 64, 64)).shape[1]
        self.train(was_training)
        return channels

    def forward(self, x):
        return self.head(self.features(x))


# --- DATASET ---
class YOLOCentroidDataset(Dataset):
    def __init__(self, split_root, num_classes, img_size=480, grid_size=60,
                 sigma=GAUSS_SIGMA):
        self.img_dir = os.path.join(split_root, "images")
        self.label_dir = os.path.join(split_root, "labels")

        if not os.path.exists(self.img_dir):
            self.img_dir = split_root
            self.label_dir = split_root.replace("images", "labels")

        self.img_paths = sorted([
            p for p in glob.glob(os.path.join(self.img_dir, "*"))
            if p.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.num_classes = num_classes
        self.grid_size = grid_size
        self.sigma = sigma

        # RIMOSSA normalizzazione ImageNet.
        # ToTensor() scala già automaticamente i pixel da 0-255 a 0.0-1.0
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        if len(self.img_paths) == 0:
            print(f"\n[!] ERRORE: Cartella immagini vuota in {self.img_dir}")
            sys.exit(1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        # Target = one soft Gaussian heatmap per class (C, grid, grid), no bg channel.
        gs = self.grid_size
        target = torch.zeros((self.num_classes, gs, gs), dtype=torch.float32)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    cls_id, x_c, y_c = int(parts[0]), float(parts[1]), float(parts[2])
                    if not (0 <= cls_id < self.num_classes):
                        continue
                    # Centre cell on the grid, then splat a Gaussian around it.
                    gx = min(int(x_c * gs), gs - 1)
                    gy = min(int(y_c * gs), gs - 1)
                    draw_gaussian(target[cls_id], gx, gy, self.sigma)

        return self.transform(img), target


def centernet_focal_loss(logits, target, eps=1e-6):
    """Penalty-reduced focal loss (CenterNet / CornerNet) on Gaussian heatmaps.

    logits: raw model output (B, C, H, W). target: Gaussian heatmap in [0,1].
    Cells where target == 1 are positives; all others are negatives whose loss
    is down-weighted by (1 - target)^4, so cells near a peak barely contribute.
    Normalized by the number of positive (peak) cells.
    """
    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pos = target.eq(1.0).float()
    neg = 1.0 - pos
    neg_weights = torch.pow(1.0 - target, 4.0)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, 2.0) * pos
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, 2.0) * neg_weights * neg

    num_pos = pos.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos


# --- TRAINING ---
def train():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(src_dir, "datasets", "split_centroid_dataset")
    yaml_path = os.path.join(dataset_root, "data.yaml")

    print("\n" + "=" * 60)
    print(f"LOCALIZZAZIONE SCRIPT: {src_dir}")
    print(f"LOCALIZZAZIONE DATASET: {dataset_root}")
    print("=" * 60 + "\n")

    if not os.path.exists(yaml_path):
        print(f"ERRORE: Manca data.yaml in {yaml_path}")
        sys.exit(1)

    with open(yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)

    train_subdir = os.path.basename(data_cfg['train'])
    val_subdir = os.path.basename(data_cfg['val'])

    train_path = os.path.join(dataset_root, train_subdir)
    val_path = os.path.join(dataset_root, val_subdir)

    # Accelerazione nativa per Apple Silicon (M1/M2/M3)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware Accelerato: {device}")

    num_classes = len(data_cfg['names'])

    train_ds = YOLOCentroidDataset(train_path, num_classes, img_size=480, grid_size=30)
    val_ds = YOLOCentroidDataset(val_path, num_classes, img_size=480, grid_size=30)

    print(f"Immagini trovate: Training={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    model = FOMO_PL_480(num_classes=num_classes).to(device)

    # Target smearing -> CenterNet penalty-reduced focal loss on sigmoid heatmaps.
    criterion = centernet_focal_loss

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 60
    for epoch in range(epochs):
        model.train()
        l_sum = 0

        print(f"\n---> Inizio Epoch {epoch + 1}/{epochs}...")

        # Aggiungiamo 'enumerate' per contare i batch
        for i, (imgs, targs) in enumerate(train_loader):
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()

            preds = model(imgs)

            if preds.shape[-2:] != targs.shape[-2:]:
                raise RuntimeError(
                    f"Grid mismatch: model output {tuple(preds.shape[-2:])} vs target "
                    f"{tuple(targs.shape[-2:])}. Set grid_size to the model's output "
                    f"resolution (a 480px input through this backbone)."
                )

            loss = criterion(preds, targs)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()

            if (i + 1) % 2 == 0:
                print(f"      Batch {i + 1}/{len(train_loader)} elaborato. Loss temporanea: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{epochs} COMPLETATA - Loss Media: {l_sum / len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(src_dir, "fomo_pl_mac.pt"))
    print("\nTraining completato. Modello salvato come fomo_pl_mac.pt")


if __name__ == "__main__":
    train()