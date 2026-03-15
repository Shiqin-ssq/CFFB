import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("/root/YOLOv11/runs/train/CFFB-yolo11/weights/last.pt")
    model.train(resume = True,
                project='runs/train',
                name='CFFB-yolo11',
                )

    # 评估模型在验证集上的性能
    metrics = model.val()

    