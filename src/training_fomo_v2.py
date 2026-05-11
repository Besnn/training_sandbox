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


class FOMO_PL_480(nn.Module):
    def __init__(self, num_classes):
        super(FOMO_PL_480, self).__init__()
        # MobileNetV2 di default.
        # NOTA: EI usa alpha=0.35, PyTorch di default usa alpha=1.0 (più grosso e preciso).
        backbone = models.mobilenet_v2(weights='DEFAULT').features

        # Tagliamo la rete esattamente al punto di risoluzione 1/8 (block_6)
        # Questo garantisce che un input 640x640 esca come 80x80 con 32 canali
        self.features = nn.Sequential(*list(backbone.children())[:7])

        # La "Testa" esatta di FOMO: Solo convoluzioni 1x1, niente 3x3, niente dropout.
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=num_classes + 1, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.head(self.features(x))


# --- DATASET ---
class YOLOCentroidDataset(Dataset):
    def __init__(self, split_root, img_size=480, grid_size=60):
        self.img_dir = os.path.join(split_root, "images")
        self.label_dir = os.path.join(split_root, "labels")

        if not os.path.exists(self.img_dir):
            self.img_dir = split_root
            self.label_dir = split_root.replace("images", "labels")

        self.img_paths = sorted([
            p for p in glob.glob(os.path.join(self.img_dir, "*"))
            if p.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.grid_size = grid_size

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

        # Il target è un tensore di indici di classe (0 = background)
        target_grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3: continue
                    cls_id, x_c, y_c = int(parts[0]), float(parts[1]), float(parts[2])

                    # Calcolo della cella nella griglia 80x80
                    gx = min(int(x_c * self.grid_size), self.grid_size - 1)
                    gy = min(int(y_c * self.grid_size), self.grid_size - 1)

                    # +1 perché lo 0 è riservato al background
                    target_grid[gy, gx] = cls_id + 1

        return self.transform(img), target_grid


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

    train_ds = YOLOCentroidDataset(train_path)
    val_ds = YOLOCentroidDataset(val_path)

    print(f"Immagini trovate: Training={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    num_classes = len(data_cfg['names'])
    model = FOMO_PL_480(num_classes=num_classes).to(device)

    # IL SEGRETO DI FOMO: Background=1.0, Oggetti=100.0
    loss_weights = torch.tensor([1.0] + [100.0] * num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

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

            loss = criterion(preds, targs)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()

            # STAMPA DI DEBUG: Ogni 10 batch (es. 10, 20, 30...)
            if (i + 1) % 2 == 0:
                print(f"      Batch {i + 1}/{len(train_loader)} elaborato. Loss temporanea: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{epochs} COMPLETATA - Loss Media: {l_sum / len(train_loader):.4f}")

    # Salva il modello per l'esportazione in ONNX successiva
    torch.save(model.state_dict(), os.path.join(src_dir, "fomo_pl_mac.pt"))
    print("\nTraining completato. Modello salvato come fomo_pl_mac.pt")


if __name__ == "__main__":
    train()