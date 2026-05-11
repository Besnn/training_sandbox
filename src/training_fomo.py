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


class FOMO_PL_640(nn.Module):
    def __init__(self, num_classes):
        super(FOMO_PL_640, self).__init__()
        backbone = models.mobilenet_v2(weights='DEFAULT').features
        self.features = nn.Sequential(*list(backbone.children())[:7])
        self.head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, num_classes + 1, kernel_size=1)
        )

    def forward(self, x):
        return self.head(self.features(x))


# --- 2. DATASET (Logica di ricerca forzata) ---
class YOLOCentroidDataset(Dataset):
    def __init__(self, split_root, img_size=640, grid_size=80):
        # Cerchiamo di capire dove sono le immagini
        # Proviamo: split_root/images o split_root direttamente
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
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if len(self.img_paths) == 0:
            print(f"\n[!] ERRORE: Cartella immagini vuota!")
            print(f"Path provato: {self.img_dir}")
            sys.exit(1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        filename = os.path.basename(img_path)
        label_name = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)
        target_grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 3: continue
                    cls_id, x_c, y_c = int(parts[0]), float(parts[1]), float(parts[2])
                    gx = min(int(x_c * self.grid_size), self.grid_size - 1)
                    gy = min(int(y_c * self.grid_size), self.grid_size - 1)
                    target_grid[gy, gx] = cls_id + 1
        return self.transform(img), target_grid


# --- 3. TRAINING ---
def train():
    # 1. Trova la cartella dove si trova questo file (/src)
    src_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Costruiamo il path del dataset (src -> datasets -> split_centroid_dataset)
    # NOTA: Uso 'split_centroid_dataset' perché è quello che appare nel tuo log SI
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

    # ESTRAZIONE PATH DAL YAML (Pulizia da eventuali ../)
    # Se il yaml dice 'train: train', os.path.basename lo pulisce
    train_subdir = os.path.basename(data_cfg['train'])
    val_subdir = os.path.basename(data_cfg['val'])

    train_path = os.path.join(dataset_root, train_subdir)
    val_path = os.path.join(dataset_root, val_subdir)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Inizializzazione Dataset con percorsi ricostruiti forzatamente
    train_ds = YOLOCentroidDataset(train_path)
    val_ds = YOLOCentroidDataset(val_path)

    print(f"Immagini trovate: Training={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    num_classes = len(data_cfg['names'])
    model = FOMO_PL_640(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1] + [1.0] * num_classes).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(30):
        model.train()
        l_sum = 0
        for imgs, targs in train_loader:
            imgs, targs = imgs.to(device), targs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), targs)
            loss.backward()
            optimizer.step()
            l_sum += loss.item()
        print(f"Epoch {epoch + 1} - Loss: {l_sum / len(train_loader):.4f}")

    torch.save(model.state_dict(), os.path.join(src_dir, "fomo_pl.pt"))


if __name__ == "__main__":
    train()