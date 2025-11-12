#将pt文件转化为onnx
#将onnx导出TensorRT引擎文件的工具
#说明：导出目录默认与源文件在同一目录

import torch
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def export_onnx(model_path,name_path,half=True):
    """将pt文件转化为onnx

    Args:
        model_path (str): 输入的.pt模型文件路径
        name_path (str, optional): 输出的ONNX文件路径（不含扩展名）
        half (bool): 是否使用FP16精度，默认为True
    """
    model = YOLO(model_path)
    # 导出模型到指定文件
    model.export(
        format='onnx',
        name=name_path,
        imgsz=640,
        simplify=False,
        dynamic=False,
        device=device,
        half = half
    )

def export_engine(model_path,name_path,half=True):
    """将YOLO模型从ONNX格式导出为TensorRT引擎格式

        Args:
            model_path (str): 输入的.onnx模型文件路径
            name_path (str, optional): 输出的ENGINE文件路径（不含扩展名）
            half (bool): 是否使用FP16精度加速推理，默认为True[3](@ref)
     """
    model = YOLO(model_path)
    # 导出模型到指定文件
    model.export(
        format='engine',
        name=name_path,
        imgsz=640,
        simplify=False,
        dynamic=False,
        device=device,
        half=half
    )

if __name__ == '__main__':
    #export_onnx('/home/pathos/桌面/Radar26/yolo11n.pt', '/home/pathos/桌面/Radar26/armor', half=True)
    export_engine('/home/pathos/桌面/Radar26/yolo11n.pt', '/home/pathos/桌面/Radar26/armor', half=True)
