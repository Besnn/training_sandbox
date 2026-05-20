from ultralytics import YOLO

model = YOLO("models/best.pt")

model.export(format="ncnn", half=True)