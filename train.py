import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(".\CFFB-yolo11.yaml")
    # model.load('yolo11n.pt')    #加载预训练权重
    model.train(data="./mydataset2/data.yaml",
                task="detect",
                cache=False,
                imgsz=640,
                epochs=30,
                #single_cls=True,  # 是否是单类别检测
                batch=32,
                close_mosaic=0,
                workers=4,
                #device='0',
                optimizer='AdamW',
                # amp=True,
                amp=True,# 小波下采样要关闭amp
                patience = 0,
                project='runs/train',
                name='exp_test',
                )

    # 评估模型在验证集上的性能
    metrics = model.val()

    