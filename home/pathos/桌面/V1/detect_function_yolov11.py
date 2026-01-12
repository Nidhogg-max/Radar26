# 导入需要的库
import os
import sys
import time
from pathlib import Path
import cv2
import random
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


class YOLOv11Detector:
    def __init__(self, weights_path, img_size=640, conf_thres=0.70, iou_thres=0.2, max_det=10,
                 device='', classes=None, augment=False, visualize=False, half=True, data='coco8.yaml', ui=False):
        """
        YOLOv11检测器初始化（包含完整预处理）
        """
        self.ui = ui
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载YOLOv11模型
        try:
            self.model = YOLO(weights_path)
            print(f"✅ YOLOv11模型加载成功: {weights_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

        # 设置模型参数
        self.model.overrides['conf'] = conf_thres
        self.model.overrides['iou'] = iou_thres
        self.model.overrides['agnostic_nms'] = False
        self.model.overrides['max_det'] = max_det
        self.model.overrides['classes'] = classes
        self.model.overrides['augment'] = augment
        self.model.overrides['verbose'] = False

        # 获取类别名称
        self.names = self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.augment = augment
        self.visualize = visualize
        self.half = half and self.device != 'cpu'

        if self.half:
            self.model.model.half()

        print(f"🎯 模型初始化完成: 设备={self.device}, 尺寸={img_size}, 半精度={self.half}")

    def _preprocess_image(self, img):
        """
        手动图像预处理（类似YOLOv5的letterbox处理）
        """
        # 保存原始图像
        im0 = img.copy()

        # 获取模型期望的输入尺寸
        if hasattr(self.model, 'model'):
            # 从模型配置获取尺寸
            model_cfg = self.model.model.args if hasattr(self.model.model, 'args') else {}
            imgsz = model_cfg.get('imgsz', self.img_size)
        else:
            imgsz = self.img_size

        # 使用letterbox进行预处理（保持宽高比的resize + padding）
        im, ratio, (dw, dh) = self.letterbox(im0, new_shape=(imgsz, imgsz), auto=False, scaleup=True)

        # BGR to RGB
        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        im = np.ascontiguousarray(im)

        # 转换为tensor并归一化
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255.0  # 归一化 0-255 to 0.0-1.0

        if len(im.shape) == 3:
            im = im.unsqueeze(0)  # 添加batch维度

        return im, im0, ratio, (dw, dh)

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # 调整图像尺寸并保持宽高比
        shape = im.shape[:2]  # 当前尺寸 [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # 只缩小不放大
            r = min(r, 1.0)

        # 计算新的未填充尺寸
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # 最小矩形
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        # 分割padding到两侧
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:  # 调整尺寸
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框

        return im, r, (dw, dh)

    def _scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """
        将坐标从预处理后的图像尺寸映射回原始图像尺寸
        """
        if ratio_pad is None:  # 计算比例
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain

        # 裁剪到图像边界内
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x轴
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y轴
        return coords

    def predict(self, img):
        """
        执行目标检测推理（包含完整预处理）
        """
        try:
            # 方法1: 使用Ultralytics内置预处理（推荐）
            results = self.model.predict(
                img,
                imgsz=self.img_size,
                conf=self.conf_thres,
                iou=self.iou_thres,
                classes=self.classes,
                max_det=self.max_det,
                augment=self.augment,
                verbose=False
            )

            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # 获取坐标 (xyxy格式)
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = xyxy
                        w, h = x2 - x1, y2 - y1
                        xywh = [int(x1), int(y1), int(w), int(h)]

                        # 获取置信度和类别
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = self.names[cls]

                        # 如果用于UI界面，绘制检测结果
                        if self.ui:
                            self._draw_detection(img, xyxy, class_name, conf, cls)

                        line = (class_name, xywh, conf)
                        detections.append(line)

            return detections

        except Exception as e:
            print(f"❌ 推理过程中出错: {e}")
            return []

    def predict_manual_preprocess(self, img):
        try:
            # 手动预处理
            im, im0, ratio, pad = self._preprocess_image(img)

            # 推理
            with torch.no_grad():
                pred = self.model.model(im)  # 直接调用模型

            # 后处理（需要根据具体模型输出格式调整）
            # 这里简化处理，实际需要根据YOLOv11的输出格式进行NMS等操作
            detections = self._process_predictions(pred, im0.shape, ratio, pad)

            return detections

        except Exception as e:
            print(f"❌ 手动预处理推理出错: {e}")
            return []

    def _process_predictions(self, pred, orig_shape, ratio, pad):
        """
        处理模型原始输出
        """
        # 这里需要根据YOLOv11的实际输出格式进行解析
        # 包括NMS、坐标映射等操作
        detections = []
        # 实现细节需要根据具体模型调整
        return detections

    def _draw_detection(self, img, xyxy, class_name, conf, cls):
        """绘制检测框和标签"""
        x1, y1, x2, y2 = map(int, xyxy)

        # 绘制边界框
        color = self.colors[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label = f'{class_name} {conf:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

        cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                      (x1 + label_size[0], y1), color, -1)

        # 绘制标签文本
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


class YOLOv11Classifier:
    def __init__(self, weights_path, img_size=224, device='', half=True):
        """YOLOv11分类器初始化"""
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载YOLOv11分类模型
        self.model = YOLO(weights_path, task='classify')  # 指定任务为分类

        # 获取类别名称
        self.names = self.model.names
        self.img_size = img_size
        self.half = half and self.device != 'cpu'

        if self.half:
            self.model.model.half()

    def predict(self, img, top_k=5):
        """
        执行图像分类
        Args:
            img: 输入图像
            top_k: 返回前k个预测结果
        Returns:
            list of (class_name, confidence)
        """
        # 使用YOLO的分类预测
        results = self.model.predict(
            img,
            imgsz=self.img_size,
            verbose=False
        )

        detections = []

        for r in results:
            if r.probs is not None:  # 分类模型有probs属性
                # 获取top_k预测
                probs = r.probs
                topk = probs.topk(top_k)  # 获取前k个

                for i in range(top_k):
                    cls_idx = topk.indices[i].item()
                    confidence = topk.values[i].item()
                    class_name = self.names[cls_idx]

                    detections.append((class_name, confidence))

        return detections