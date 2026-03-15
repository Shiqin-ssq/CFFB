
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO('best.pt')
    model.predict(
                  source='E:/python/YOLO/improve/YOLOv11/mydataset2/test/images',
                  save=True,
                  project='runs/test',
                  name='enhanced_underwater',
                  )
