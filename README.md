# Radar26 Final · 面向 RoboMaster 2026 赛季的全自动雷达视觉与决策辅助系统

![Platform](https://img.shields.io/badge/platform-Linux--x86__64-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-competition--ready-red)

> 基于海康工业相机（兼容 USB 相机）的 RoboMaster 雷达站上位机系统。  
> 通过 **YOLOv11 三层级联检测** → **卡尔曼滤波 + 匈牙利算法跟踪** → **多高度层坐标映射**，实现对敌方机器人 **实时定位、ID 识别与运动预测**，并通过串口将战场态势上报给决策模块。  
> 支持 **TensorRT 引擎加速**、**盲区智能预测**、**地图掩码编辑器**及 **双阵营适配**（红/蓝方），适用于 **RoboMaster 2026 赛季** 的雷达站开发与竞赛场景。

---

## 📖 目录

1. [系统特性](#系统特性)  
2. [视觉检测与定位流水线](#视觉检测与定位流水线)  
3. [项目结构](#项目结构)  
4. [环境与依赖](#环境与依赖)  
5. [快速开始](#快速开始)  
   - [5.1 安装基础依赖](#51-安装基础依赖)  
   - [5.2 准备模型与资源文件](#52-准备模型与资源文件)  
   - [5.3 修改主配置文件](#53-修改主配置文件)  
   - [5.4 启动系统](#54-启动系统)  
6. [相机标定与多高度层映射](#相机标定与多高度层映射)  
7. [模型训练与 TensorRT 导出](#模型训练与-tensorrt-导出)  
8. [辅助工具与脚本](#辅助工具与脚本)  
9. [串口通信协议](#串口通信协议)  
10. [运行截图与调试](#运行截图与调试)  
11. [许可证](#许可证)  
12. [致谢](#致谢)  

---

## 系统特性

- **三层级联神经网络**  
  `车体检测(1280×1280)` → `装甲板检测(192×192)` → `数字分类(64×64)`，由粗到细快速推断敌方机器人完整 ID（如 R1、B3、R7 等）。

- **硬件加速推理**  
  使用 TensorRT FP16 静态引擎，在 NVIDIA Jetson / 独立 GPU 上稳定达到 **60 FPS 以上** 的检测速度。

- **鲁棒的多目标跟踪**  
  自研 2D 卡尔曼滤波器（位置 + 速度）与匈牙利算法匹配检测框，可应对短暂遮挡、漏检等场景，并在长时未观测后自动清除丢失目标。

- **多高度层坐标变换**  
  利用掩码图像自动区分地面、R 型高地、环形高地，并加载三组透视变换矩阵，将图像下边缘中心点精确映射至 **2800×1500** 的地图坐标系。

- **盲区预测**  
  基于历史轨迹的速度方向与预设盲区点位，对敌方消失后的可能位置进行实时评分排序，提升决策的持续性。

- **全自动串口通信**  
  定时打包整场敌方坐标发送给下位机，同时接收裁判系统返回的进度条、双倍易伤、飞镖目标等信息，并在 UI 动态展示。

- **可视化调试窗口**  
  实时显示：① 原始图像 + 检测框 + 数字识别结果；② 战场地图 + 敌方位置 + 速度矢量；③ 裁判系统血量/进度条面板。支持按键截图、暂停、重置跟踪器等操作。

- **地图掩码编辑器**  
  基于 PyQt5 的图形化工具，可直接在 2800×1500 地图上绘制多边形填充区域（高度层），并导出为掩码图片供标定使用。

---

## 视觉检测与定位流水线

```mermaid
graph TD
    A[海康/USB相机采集] --> B[车体检测器<br/>YOLOv11-1280]
    B -->|裁剪ROI| C[装甲板检测器<br/>YOLOv11-192]
    C -->|二次裁剪| D[数字分类器<br/>YOLOv11-64]
    D --> E[拼接完整ID<br/>R1/B3...]
    E --> F{依据掩码选择<br/>高度层变换矩阵}
    F -->|地面| G1[地面矩阵 M_GROUND]
    F -->|R型高地| G2[红色矩阵 M_HEIGHT_R]
    F -->|环形高地| G3[绿色矩阵 M_HEIGHT_G]
    G1 & G2 & G3 --> H[得到地图坐标 (x_map, y_map)]
    H --> I[卡尔曼滤波更新 + 匈牙利匹配]
    I --> J[地图显示 + 盲区预测]
    I --> K[串口打包坐标发送]
---

## 项目结构

---

Radar26_Final/
├── V1/                           # 主工作目录
│   ├── main.py                   # ★ 主程序入口
│   ├── detect_function_yolov11.py # 检测器/分类器封装 (YOLOv11)
│   ├── hik_camera.py             # 海康相机SDK封装 (Windows/Linux)
│   ├── information_ui.py         # 裁判系统进度条绘制
│   ├── guess_plt.py              # 盲区预测算法
│   ├── train.py                  # 模型训练脚本
│   ├── onnx2engine.py            # ONNX→TensorRT引擎导出
│   ├── calibration.py            # 旧版标定UI (基于PyQt5)
│   ├── biao.py                   # 9点标定矩阵计算
│   ├── check.py                  # 标定点对可视化检查
│   ├── map.py                    # 纯色地图生成工具
│   ├── PNG_draw.py               # 地图掩码编辑器 (PyQt5)
│   ├── devide.py                 # 数据集划分脚本
│   ├── teat.py                   # 离线视频测试工具
│   ├── NEXT-E_axis.py            # 盲区点位可视化
│   ├── modelEEE/                 # ⚠️ 需自行准备模型文件
│   │   ├── car_1280_best.engine  #   车体检测 TensorRT引擎
│   │   ├── armorm_192_best.engine#   装甲板检测 TensorRT引擎
│   │   ├── cls_64_best.engine    #   数字分类 TensorRT引擎
│   │   └── *.pt / *.onnx         #   训练中间产物
│   ├── images/                   # 地图与掩码资源
│   │   ├── 2025map.png           # 战场地图 (2800x1500)
│   │   ├── 2025map_red.png       # 红方视角地图
│   │   ├── 2025map_blue.png      # 蓝方视角地图
│   │   ├── 2025map_mask.png      # 蓝方高度层掩码
│   │   ├── map_mask.jpg          # 红方高度层掩码
│   │   └── test.mp4 / test.jpg   # 测试素材
│   ├── arrays_test_red.npy       # 红方三维标定矩阵
│   ├── arrays_test_blue.npy      # 蓝方三维标定矩阵
│   ├── arrays_test.npy           # 地面层矩阵
│   ├── yaml/                     # 数据集配置文件
│   │   ├── car.yaml
│   │   ├── armor.yaml
│   │   └── armor_classify.yaml
│   ├── RM_serial_py/             # 串口通信子模块
│   ├── MvImport_Linux/           # 海康相机 Linux SDK 封装
│   └── MvImport/                 # 海康相机 Windows SDK 封装
└── README.md                     # 本文件
