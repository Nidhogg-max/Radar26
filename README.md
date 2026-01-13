# RoboMaster 雷达识别与定位系统

## 主要功能

### 1. 三级目标检测
- **第一级：车体检测** - 识别战场上的所有机器人车体
- **第二级：装甲板检测** - 在车体区域内识别装甲板
- **第三级：数字分类** - 对车体种类进行区分

### 2. 多高度层坐标转换
- **地面层** - 黑色掩码区域
- **R型高地** - 绿色掩码区域（400mm高度）
- **环形高地** - 蓝色掩码区域（600mm高度）

### 3. 高级跟踪算法
- **匈牙利算法** - 用于数据关联，匹配检测框与跟踪器
- **卡尔曼滤波** - 用于状态估计，提供平滑的位置和速度信息

### 4. 实时状态显示
- 战场地图上实时显示敌方机器人位置
- 显示标记进度、双倍易伤状态
- 显示FPS、跟踪器数量等系统信息

### 5. 串口通信
- 发送机器人坐标到裁判系统
- 接收裁判系统的状态信息（标记进度、双倍易伤等）

## 系统架构
输入层：相机图像 → 检测层：YOLOv11检测 → 坐标转换：仿射变换
↓
数据处理：卡尔曼滤波 → 数据关联：匈牙利算法 → 输出层：串口通信
↓
状态显示：地图绘制 → UI显示：信息面板 → 视频录制：可选录制


## 主要逻辑流程

### 1. 图像获取
- 支持海康相机、USB相机和测试模式
- 多线程获取，避免阻塞主程序

### 2. 目标检测及定位流程
```python
# 第一级：车体检测
detections = detector.predict(image)  # 检测所有车体

# 第二级：装甲板检测
for car_detection in detections:
    if cls == 'car':
        # 裁剪车体区域
        cropped = image[top:top+h, left:left+w]
        # 检测装甲板
        armor_detections = detector_next.predict(cropped)

# 第三级：数字分类
if result2:  
    armor_class, class_conf = result2[0]
    ......  

# 三层仿射变换
# 1. 地面层转换
ground_point = perspectiveTransform(point, M_ground)

# 2. 通过掩码颜色判断高度
if mask_color == black:      # 地面层
    return ground_point
elif mask_color == green:    # R型高地
    return perspectiveTransform(point, M_height_r)
elif mask_color == blue:     # 环形高地
    return perspectiveTransform(point, M_height_g)

# 卡尔曼滤波预测
predicted_positions = []
for tracker in trackers:
    tracker.predict()  # 卡尔曼滤波预测
    
# 匈牙利算法匹配
matches = HungarianAlgorithm.match(detections, trackers)

# 更新跟踪器
for det_idx, trk_idx in matches:
    trackers[trk_idx].update(detections[det_idx])


STATE = 'R'  # 阵营：'R'红方 / 'B'蓝方
USART = 1    # 串口开关：0=关闭，1=开启
USER_COM = 'COM8'  # 串口号，Linux系统可能是 '/dev/ttyUSB0'

USER_MODE = 'test'  # 可选：'test'测试模式 / 'hik'海康相机 / 'video'USB相机
USER_IMG_TEST = 'images/test_image.jpg'  # 测试模式下的图像路径

SAVE_IMG = 0        # 视频录制：0=关闭，1=开启
GAME_DIR = "tete"   # 视频保存目录名称

conf_thres = 0.1    # 车体检测置信度阈值（0-1，值越大要求越严格）
iou_thres = 0.5     # 车体检测IoU阈值（0-1，值越大要求越严格）
conf_thres = 0.4    # 装甲板检测置信度阈值
iou_thres = 0.2     # 装甲板检测IoU阈值

iou_threshold = 0.3          # 匈牙利算法匹配阈值（0-1）
process_noise_std = 0.1      # 卡尔曼滤波过程噪声标准差
measurement_noise_std = 1.0  # 卡尔曼滤波测量噪声标准差
max_miss_before_removal = 15 # 连续丢失多少次后移除跟踪器

RM_serial_py/                  # 串口通信模块（如需串口功能）
save_video/                   # 视频保存目录（如需录制）
