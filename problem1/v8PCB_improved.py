from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8_improved.yaml')
    model.train(data="PCB.yaml", imgsz=640, batch=16, workers=8, cache=True, epochs=20)
    metrics = model.val()
    path = model.export(format="onnx", opset=13)
