from ultralytics import YOLO

model = YOLO('models/best.pt')

success = model.export(format='onnx', imgsz=640, opset=21)

if success:
    print("Conversione completata! Il file è 'best.onnx'")