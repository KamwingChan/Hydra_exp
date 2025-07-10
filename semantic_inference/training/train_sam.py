from ultralytics import YOLO

# 加载 YOLOv8 预训练的分割模型
model = YOLO('yolov8n-seg.pt')

# 开始训练
results = model.train(
    data='training/ade20k_annotated.yaml',
    epochs=100,
    imgsz=640
)
