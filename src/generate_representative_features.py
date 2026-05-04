import cv2
import numpy as np
import glob
import os


IMAGE_PATH = "datasets/split_obb_dataset/train/images/*.jpg"
IMG_SIZE = 640
OUTPUT_FILE = "etc/representative_features.npy"
NUM_SAMPLES = 100


def generate_features():
    img_list = glob.glob(IMAGE_PATH)
    if not img_list:
        print(f"Errore: Nessuna immagine trovata in {IMAGE_PATH}")
        return

    representative_data = []
    print(f"Elaborazione di {min(NUM_SAMPLES, len(img_list))} immagini...")

    for i in range(min(NUM_SAMPLES, len(img_list))):
        img = cv2.imread(img_list[i])
        if img is None: continue

        # Pre-processing identico a YOLOv8
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Inverte BGR -> RGB
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Ridimensiona
        img = img.astype(np.float32) / 255.0  # Normalizza 0-1

        representative_data.append(img)

    # Trasforma in un array NumPy con shape (N, 640, 640, 3)
    final_array = np.array(representative_data, dtype=np.float32)

    np.save(OUTPUT_FILE, final_array)
    print(f"File salvato con successo: {OUTPUT_FILE}")
    print(f"Shape finale: {final_array.shape}")  # Dovrebbe essere (100, 640, 640, 3)


if __name__ == "__main__":
    generate_features()