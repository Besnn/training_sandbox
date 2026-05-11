import os
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import glob

MODEL_PATH = "models/best.pt"
DATASET_PATH = "datasets/split_obb_dataset/train/images/*.jpg"
IMG_SIZE = 640
OUTPUT_FILENAME = "models/model_full_int8.tflite"
NUM_CALIBRATION_IMAGES = 100


def representative_dataset_gen():
    img_list = glob.glob(DATASET_PATH)
    if not img_list:
        raise ValueError(f"Nessuna immagine trovata in {DATASET_PATH}")

    samples = img_list[:NUM_CALIBRATION_IMAGES]

    for p in samples:
        img = cv2.imread(p)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Normalizzazione 0.0 - 1.0
        img = img.astype(np.float32) / 255.0

        # Aggiunta della dimensione batch (1, 640, 640, 3)
        img = np.expand_dims(img, axis=0)
        yield [img]


def convert_to_full_int8():
    print("Esportazione in formato SavedModel in corso...")
    model = YOLO(MODEL_PATH)
    model.export(format='tflite', imgsz=IMG_SIZE, int8=False)

    saved_model_path = MODEL_PATH.replace('.pt', '_saved_model')

    print("Inizio Full Integer Quantization forzata...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # FORZATURA: Impone l'uso esclusivo di operazioni intere
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # FORZATURA: Converte anche l'input e l'output in INT8 (fondamentale per la NPU)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Calibrazione con il dataset rappresentativo
    converter.representative_dataset = representative_dataset_gen

    try:
        tflite_model = converter.convert()
        with open(OUTPUT_FILENAME, "wb") as f:
            f.write(tflite_model)
        print(f"Conversione completata con successo: {OUTPUT_FILENAME}")
    except Exception as e:
        print(f"Errore critico durante la conversione: {e}")
        print("Nota: Se ricevi errori su operatori non supportati, il modello OBB potrebbe")
        print("contenere funzioni matematiche non convertibili in puro INT8.")


if __name__ == "__main__":
    convert_to_full_int8()