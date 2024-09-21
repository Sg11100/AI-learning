from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
# 开始训练，使用 COCO128 数据集
    model.train(data='coco128.yaml', imgsz=640, batch=16, epochs=10)  # train the model
# 使用验证集验证模型性能
    metrics = model.val()
# 对新图像进行预测
    results = model.predict(source=r"D:\code\yolo\datasets\coco128\images\train2017\000000000257.jpg", save=True, imgsz=640)



