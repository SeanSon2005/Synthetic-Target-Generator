from ultralytics import YOLO

model = YOLO('best.pt')

model.export(format='onnx',half=True,simplify=True,imgsz=640)