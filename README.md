<<<<<<< HEAD
# 雷达视觉识别与定位系统

基于 YOLOv11 和计算机视觉技术的实时识别与精确定位系统。

 系统功能

 **双阶段目标识别**：车辆检测 + 装甲板识别
 **多级精确定位**：仿射变换 + PnP算法 + 卡尔曼滤波  
 **实时数据显示**：战场地图 + 识别结果 + 系统状态
 **串口通信**：与裁判系统实时数据交互
 **盲区预测**：智能位置预测

 完整工作流程

 #  机器人视觉识别与定位系统

基于 YOLOv11 和计算机视觉技术的实时识别与精确定位系统。

### 第一阶段：目标识别

相机图像 → YOLOv11检测 → 获取ROI区域 → YOLOv11装甲板检测 → 精确边界框

### 第二阶段：多级定位  

装甲板坐标 → 仿射变换粗略定位 → PnP算法精确3D定位 → 卡尔曼滤波运动平滑 → 最终地图坐标

### 第三阶段：输出显示

定位结果 → UI实时显示 + 串口数据发送 → 裁判系统

### 环境要求
```bash
# 核心依赖
pip install opencv-python numpy pyserial torch torchvision
# GPU加速（可选）
pip install tensorrt pycuda

项目根目录/
├── main.py                 # 主程序
├── models/                 # 模型文件
│   ├── car.onnx           # 车辆检测模型
│   └── armor.onnx         # 装甲板检测模型
├── images/                 # 图像资源
│   ├── map.jpg            # 战场地图
│   ├── map_mask.jpg       # 地图掩码
│   └── test_image.jpg     # 测试图像
├── yaml/                  # 配置文件
│   ├── car.yaml           # 车辆检测配置
│   └── armor.yaml         # 装甲板检测配置
└── 标定文件/               # 定位标定
    ├── arrays_test_red.npy   # 红方仿射矩阵
    └── arrays_test_blue.npy  # 蓝方仿射矩阵

基本配置
# 在 main.py 中修改以下参数：
state = 'B'              # 阵营：'R'红方 / 'B'蓝方
user_mode = 'test'       # 模式：'test'测试 / 'hik'海康相机 / 'video'USB相机  
user_com = 'COM19'       # 串口号


显示窗口
系统运行后显示三个窗口：
information_ui 战场信息面板（进度、易伤状态）
map 战场地图（显示敌方机器人位置）
img 原始图像（叠加识别框）

图像尺寸: (1080, 1920, 3)
检测到机器人: car, 置信度: 0.89
检测到装甲板: B1, 置信度: 0.76
定位结果  B1: 方法=PnP, 高度层=ground, 位置=(1250.3, 680.5)
fps: 25.3

# 需要根据实际相机标定修改：
camera_matrix = np.array([...])  # 相机内参矩阵
dist_coeffs = np.array([...])    # 畸变系数
armor_width = 0.13    # 装甲板实际宽度（米）
armor_height = 0.055  # 装甲板实际高度（米）

# 可根据实际场景调整：
window_size = 5           # 滑动窗口大小
max_inactive_time = 2.0   # 最大失活时间
conf_thres = 0.1          # 检测置信度阈值


算法架构
识别系统
├── YOLOv11车辆检测
└── YOLOv11装甲板检测

定位系统  
├── 仿射变换（粗略定位）
├── PnP算法（精确3D定位）
└── 扩展卡尔曼滤波（运动平滑）

=======
# 雷达视觉识别与定位系统

基于 YOLOv11 和计算机视觉技术的实时识别与精确定位系统。

 系统功能

 **双阶段目标识别**：车辆检测 + 装甲板识别
 **多级精确定位**：仿射变换 + PnP算法 + 卡尔曼滤波  
 **实时数据显示**：战场地图 + 识别结果 + 系统状态
 **串口通信**：与裁判系统实时数据交互
 **盲区预测**：智能位置预测

 完整工作流程

 #  机器人视觉识别与定位系统

基于 YOLOv11 和计算机视觉技术的实时识别与精确定位系统。

### 第一阶段：目标识别

相机图像 → YOLOv11检测 → 获取ROI区域 → YOLOv11装甲板检测 → 精确边界框

### 第二阶段：多级定位  

装甲板坐标 → 仿射变换粗略定位 → PnP算法精确3D定位 → 卡尔曼滤波运动平滑 → 最终地图坐标

### 第三阶段：输出显示

定位结果 → UI实时显示 + 串口数据发送 → 裁判系统

### 环境要求
```bash
# 核心依赖
pip install opencv-python numpy pyserial torch torchvision
# GPU加速（可选）
pip install tensorrt pycuda

项目根目录/
├── main.py                 # 主程序
├── models/                 # 模型文件
│   ├── car.onnx           # 车辆检测模型
│   └── armor.onnx         # 装甲板检测模型
├── images/                 # 图像资源
│   ├── map.jpg            # 战场地图
│   ├── map_mask.jpg       # 地图掩码
│   └── test_image.jpg     # 测试图像
├── yaml/                  # 配置文件
│   ├── car.yaml           # 车辆检测配置
│   └── armor.yaml         # 装甲板检测配置
└── 标定文件/               # 定位标定
    ├── arrays_test_red.npy   # 红方仿射矩阵
    └── arrays_test_blue.npy  # 蓝方仿射矩阵

基本配置
# 在 main.py 中修改以下参数：
state = 'B'              # 阵营：'R'红方 / 'B'蓝方
user_mode = 'test'       # 模式：'test'测试 / 'hik'海康相机 / 'video'USB相机  
user_com = 'COM19'       # 串口号


显示窗口
系统运行后显示三个窗口：
information_ui 战场信息面板（进度、易伤状态）
map 战场地图（显示敌方机器人位置）
img 原始图像（叠加识别框）

图像尺寸: (1080, 1920, 3)
检测到机器人: car, 置信度: 0.89
检测到装甲板: B1, 置信度: 0.76
定位结果  B1: 方法=PnP, 高度层=ground, 位置=(1250.3, 680.5)
fps: 25.3

# 需要根据实际相机标定修改：
camera_matrix = np.array([...])  # 相机内参矩阵
dist_coeffs = np.array([...])    # 畸变系数
armor_width = 0.13    # 装甲板实际宽度（米）
armor_height = 0.055  # 装甲板实际高度（米）

# 可根据实际场景调整：
window_size = 5           # 滑动窗口大小
max_inactive_time = 2.0   # 最大失活时间
conf_thres = 0.1          # 检测置信度阈值


算法架构
识别系统
├── YOLOv11车辆检测
└── YOLOv11装甲板检测

定位系统  
├── 仿射变换（粗略定位）
├── PnP算法（精确3D定位）
└── 扩展卡尔曼滤波（运动平滑）

>>>>>>> 905a8edf1092c706dcc4b27ae1bf5d2518d75d23
