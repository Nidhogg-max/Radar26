# export_fixed.py
import torch
from ultralytics import YOLO
import os

# 强制指定设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def export_engine_optimized(model_path, half=True):
    """
    优化版 TensorRT 导出函数
    加入 simplify 和 opset 约束，减少坐标偏移 Bug
    """
    print(f"🔄 正在处理: {model_path} ...")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 文件不存在: {model_path}")
        return

    try:
        model = YOLO(model_path)

        # 导出模型
        # 注意：name参数在某些版本会被忽略，它通常会在原目录下生成 .engine 文件
        output_name = model.export(
            format='engine',
            imgsz=64,  # 必须固定 640
            simplify=True,  # 关键修改：开启简化 (Fixes many bugs)
            opset=12,  # 关键修改：锁定 opset 版本 (Stable)
            dynamic=False,  # 静态尺寸，推理最快
            device=0,  # 指定 GPU 0
            half=half,  # FP16 半精度
            workspace=6  # 4GB 构建工作区
        )
        print(f"✅ 导出成功: {output_name}")

    except Exception as e:
        print(f"❌ 导出失败 {model_path}: {e}")


if __name__ == '__main__':
    # 请根据您的实际路径修改
    pt_files = [
        '/home/pathos/桌面/V1(1625.19)/V1/modelEEE/cls_64_best.pt',
    ]

    for pt_file in pt_files:
        export_engine_optimized(pt_file, half=True)