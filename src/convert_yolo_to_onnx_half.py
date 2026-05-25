from ultralytics import YOLO
from pathlib import Path

CUSTOM_MODEL_NAME = "models/yolo26n-obb-onnx/yolo26n-obb-fp16.onnx"

model = YOLO('models/yolov8n-obb.pt')

exported_path = model.export(format='onnx', imgsz=640, opset=21, half=True)

if exported_path:
    Path(exported_path).rename(CUSTOM_MODEL_NAME)
    print(f"Conversione completata! Il file è stato rinominato in '{CUSTOM_MODEL_NAME}'")
else:
    print("Errore durante l'esportazione del modello.")