from ultralytics import YOLO

model = YOLO('best.pt')

model.export(
    format='tflite',
    int8=True,
    data='split_dataset/data.yaml',
    imgsz=640,
    format_options={'fully_quantize': True}
)