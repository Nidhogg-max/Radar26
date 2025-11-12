#导入库
import os
import sys
import time
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()#FILE记录的是当前文件的绝对地址
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))#通过将上一级目录放入sys.path，将YOLOv5放入sys.path
ROOT = Path(os.path.relpath(ROOT,Path.cwd()))

# 导入letterbox

from ultralytics.utils.general import (LOGGER, check_img_size, scale_boxes)
import random
import torch
import numpy as np
from ultralytics.utils.general import non_max_suppression, xyxy2xywh
#non_max_suppression过滤重叠框
# xyxy2xywh，将框格式从角点坐标转为中心点坐标。
from ultralytics.utils.torch_utils import select_device
#select_device 智能原则运算设备
from ultralytics.utils.plots import Annotator
#Annotator 结果可视化，在图像上绘制检测框和标签。
from ultralytics.models.common import DetectMultiBackend
#DetectMultiBackend 加载预训练模型，支持多种格式和推理后端。
from ultralytics.utils.augmentations import letterbox
#letterbox 图像预处理，保持原图比例调整尺寸并填充。

class YOLOv5Detector:
    def __init__(self,weights_path, img_size=(640, 640), conf_thres=0.70, iou_thres=0.2, max_det=10,
                 device='', classes=None, agnostic_nms=False, augment=False, visualize=False,
                 half=True, dnn=False,
                 data='data/coco128.yaml', ui=False):
        #设置运算设备
        self.ui = ui#记录用户是否希望在推理过程中实时看到带有检测框的可视化结果
        self.device = select_device()

        #加载模型
        #dnn：是否使用 OpenCV DNN作为 ONNX 模型的后端（而非 ONNX Runtime）
        #data：	数据集配置文件路径（如 coco.yaml），用于获取类别名称
        #fp16:是否使用 FP16 半精度推理（可提升 GPU 推理速度）
        self.model = DetectMultiBackend(weights_path,device = self.device,dnn=dnn,data=data,fp16=half)

        #从已加载的模型后端中提取关键的元数据属性
        stride, self.names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        #stride：图形的下采样步长
        #pt, jit, onnx, engine ：到底模型是哪一种格式
        #self.names：类别名称字典

        #计算在经过模型下采样的图片大小
        self.img_size = check_img_size(img_size, s=stride)

        #每个可检测的类别动态生成一个随机的、唯一的颜色。
        self.colors = [[random.randint(0,225)]for _ in range(3)for _ in self.names]

        #是否开启半精度推理
        #条件：我在初始化函数是half为True，且格式为pt, jit, onnx, engine中的一个，且模型运算设备为CPU
        self.half = half and (pt or jit or onnx or engine) and self.device.type != 'cpu'
        if pt or jit:
            self.model.model.half() if self.half else self.model.model.float()

        #规定一些值
        self.save_time = 0#	用于记录最后保存结果的时间戳，通常用于控制保存频率
        self.conf_thres = conf_thres#置信度阈值，过滤掉置信度低于此值的检测框
        self.iou_thres = iou_thres#	IoU阈值，用于非极大值抑制(NMS)时判断框是否重叠
        self.max_det = max_det#	最大检测数量，限制每张图像输出的检测框数量
        self.classes = classes#	类别过滤，只检测指定类别的目标
        self.agnostic_nms = agnostic_nms#类别无关NMS，True时NMS跨所有类别进行
        self.augment = augment#测试时增强，是否使用TTA提升检测精度
        self.visualize = visualize#特征可视化，是否输出中间特征图

        bs = 1;
        # 开始预测
        #正式开始处理实际数据前，用虚拟数据先运行一次模型，以完成初始化，从而避免首次推理时因各种初始化操作
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *self.img_size))

    def predict(self,img):
        #图片预处理
        #经图片深拷贝给img0
        im0 = img.copy()
        #将任意尺寸的输入图像 im0调整为模型所需的固定输入尺寸
        im = letterbox(im0, self.img_size, self.model.stride, auto=self.model.pt)[0]
        #HWC to CHW, BGR to RGB
        im = im.transpose((2, 0, 1))[::-1]
        #确保数组 im在内存中以连续的方式存储是numpy数组的格式
        im = np.ascontiguousarray(im)
        #将 NumPy 数组 im转换为 PyTorch 张量
        im = torch.from_numpy(im).to(self.device)
        #将输入图像张量的数据类型转换为半精度（FP16）或单精度（FP32）
        im = im.half() if self.half else im.float()
        #归一化
        im /= 255
        #维度拓展
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        #预测
        pred = self.model(im, augment=self.augment, visualize=self.visualize)

        # NMS
        #过滤掉大量重叠的检测框，只保留最可能代表真实目标的少数最佳检测结果
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)

        # 用于存放结果
        detections = []

        #处理预测的结果
        for i,det in  enumerate(pred):
            if len(det):
                #映射会原始尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                #逆序遍历当前图片的所有检测框
                for *xyxy, conf, cls in reversed(det):
                    #将边界框格式从 (x1, y1, x2, y2)转换为 (x_center, y_center, width, height)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    #坐标取整
                    xywh = [round(x) for x in xywh]
                    #将中心点坐标格式转换为左上角坐标格式
                    xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
                    #是否绘制图形化界面
                    if self.ui:
                        #创建用于在图像上绘制检测结果的标注器
                        annotator = Annotator(np.ascontiguousarray(img), line_width=3, example=str(self.names))
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=self.colors[int(cls)])
                    #将识别信息添加到里line,再将line添加到detections
                    cls = self.names[int(cls)]
                    conf = float(conf)
                    line = (cls, xywh, conf)
                    detections.append(line)
        return detections