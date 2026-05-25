import os
from ultralytics import YOLO

LOCAL_DATA_YAML = "datasets/split_obb_dataset/data.yaml"


def train_obb_nano():
    """
    Allena YOLOv8 Nano con Oriented Bounding Boxes (OBB)
    """
    print("Inizializzazione Training OBB Nano...")

    model = YOLO('yolo26n-obb.pt')

    results = model.train(
        data=LOCAL_DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device="mps",  # Usa il chip Apple Silicon (M1/M2/M3)
        save=True,
        patience=20,  # Early stopping un po' più permissivo
        project='runs/obb',
        name='yolo26_nano_obb'
    )

    print("Training completato. I pesi sono in: runs/obb/yolo26_nano_obb/weights/best.pt")
    return model


if __name__ == "__main__":
    if os.path.exists(LOCAL_DATA_YAML):
        train_obb_nano()
    else:
        print(f"Errore: Non trovo il file {LOCAL_DATA_YAML}")