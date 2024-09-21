from ultralytics import YOLO
if __name__ == '__main__':

    model = YOLO('yolov9t.pt')
# 开始训练，使用 COCO128 数据集
    model.train(data='coco128.yaml', epochs=80, imgsz=640, batch=16, name='yolov8_coco128')
# 使用验证集验证模型性能
    metrics = model.val()
    print(metrics)
# 对新图像进行预测
    results = model.predict(source=r"D:\code\yolo\datasets\coco128\images\train2017\000000000404.jpg", save=True, imgsz=640)

