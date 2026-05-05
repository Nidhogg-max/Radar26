import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # 加载模型和预训练权重
    model = YOLO(model=r'/home/pathos/桌面/Radar26/ultralytics/cfg/models/11/yolo11.yaml')
    model.load('yolo11n.pt')  # 启用预训练权重

    model.train(
        data=r'/Radar26/yaml/car.yaml',
        imgsz=640,
        epochs=400,  # 增加训练轮次
        batch=32,  # 根据GPU内存调整
        workers=4,  # 适当增加数据加载线程
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer='SGD',
        lr0=0.01,  # SGD学习率
        momentum=0.937,  # 动量参数
        weight_decay=0.0005,  # 权重衰减
        cos_lr=True,  # 余弦学习率调度
        warmup_epochs=3,  # 学习率预热
        close_mosaic=10,
        resume=False,
        project='/home/pathos/桌面/Radar26',
        name='car_delect',
        single_cls=False,
        cache=False,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )