from ultralytics import YOLO

LOAD_FROM_PREV = False

if LOAD_FROM_PREV:
    model = YOLO('yolov8n.yaml').load('best.pt')
else:
    model = YOLO('yolov8n.yaml')

model.train(data='data/data.yaml', epochs=30, imgsz=640)
