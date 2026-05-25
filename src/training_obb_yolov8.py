import os
from ultralytics import YOLO

LOCAL_DATA_YAML = "datasets/split_obb_dataset/data.yaml"


def train_obb_nano():
    """
    Allena YOLOv8 Nano con Oriented Bounding Boxes (OBB)
    """
    print("Inizializzazione Training OBB Nano...")

    model = YOLO('models/yolov8n-obb.pt')

    results = model.train(
        data=LOCAL_DATA_YAML,
        epochs=100,
        imgsz=640,
        batch=16,
        device="mps",  # Usa il chip Apple Silicon (M1/M2/M3)
        save=True,
        patience=30,  # Early stopping un po' più permissivo
        project='runs/obb',
        name='train_nano_obb'
    )

    print("Training completato. I pesi sono in: runs/obb/train_yolov8_nano_obb/weights/best.pt")
    return model


if __name__ == "__main__":
    if os.path.exists(LOCAL_DATA_YAML):
        train_obb_nano()
    else:
        print(f"Errore: Non trovo il file {LOCAL_DATA_YAML}")