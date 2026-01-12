"""
RoboMaster 视觉识别系统  主程序
集成匈牙利算法（数据关联）和卡尔曼滤波（状态估计）
"""

import math
import threading
import time
import datetime
import os
import sys
from collections import deque
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import serial

# 导入项目模块 - 明确指定导入内容
try:
    # 导入信息UI模块
    import information_ui
    draw_information_ui = information_ui.draw_information_ui
    
    # 导入检测模块
    from detect_function_yolov11 import YOLOv11Detector
    
    # 导入串口模块
    from RM_serial_py.ser_api import (
        build_send_packet, 
        receive_packet, 
        Radar_decision,
        build_data_decision, 
        build_data_radar_all, 
        build_data_sentry
    )
    SERIAL_MODULE_AVAILABLE = True
    print("串口模块加载成功")
except ImportError as e:
    print(f"模块导入警告: {e}")
    # 创建模拟函数以避免运行时错误
    def draw_information_ui(*args, **kwargs):
        return [0, 0, 0, 0, 0, 0]
    
    class YOLOv11Detector:
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, img):
            return []
    
    SERIAL_MODULE_AVAILABLE = False
    print("部分模块不可用，程序将以模拟模式运行")

# ==================== 匈牙利算法实现 ====================
class HungarianAlgorithm:
    """匈牙利算法实现数据关联（检测框与跟踪器匹配）"""
    
    @staticmethod
    def compute_iou(box1: Tuple[float, float, float, float], 
                   box2: Tuple[float, float, float, float]) -> float:
        """
        计算两个边界框的IoU(交并比)
        box: (x, y, w, h)
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # 计算相交区域
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(x1 + w1, x2 + w2)
        inter_y2 = min(y1 + h1, y2 + h2)
        
        # 检查是否有相交
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        # 计算相交面积
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 计算并集面积
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou
    
    @staticmethod
    def match_detections_to_tracks(detections: List[Tuple[str, Tuple, float]],
                                  tracks: List[Any],
                                  iou_threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        使用匈牙利算法匹配检测结果到跟踪器
        
        参数:
            detections: 检测结果列表 [(cls, (x,y,w,h), conf), ...]
            tracks: 跟踪器列表
            iou_threshold: IoU阈值
            
        返回:
            匹配对列表 [(detection_idx, track_idx), ...]
        """
        if not detections or not tracks:
            return []
        
        # 构建成本矩阵（IoU越大，成本越小）
        n_det = len(detections)
        n_trk = len(tracks)
        
        # 初始化成本矩阵
        cost_matrix = np.zeros((n_det, n_trk))
        
        for i, det in enumerate(detections):
            _, det_box, _ = det
            for j, track in enumerate(tracks):
                # 获取跟踪器的预测框
                track_box = track.get_predicted_box()
                if track_box:
                    # 使用1-IoU作为成本（IoU越大，成本越小）
                    iou = HungarianAlgorithm.compute_iou(det_box, track_box)
                    cost_matrix[i, j] = 1.0 - iou
                else:
                    cost_matrix[i, j] = 1.0  # 没有预测框，成本设为最大
        
        # 简单的贪婪匹配（简化版匈牙利算法）
        matches = []
        used_detections = [False] * n_det
        used_tracks = [False] * n_trk
        
        # 按成本排序所有可能的匹配
        all_possible_matches = []
        for i in range(n_det):
            for j in range(n_trk):
                iou = 1.0 - cost_matrix[i, j]
                if iou >= iou_threshold:
                    all_possible_matches.append((iou, i, j))
        
        # 按IoU降序排序
        all_possible_matches.sort(reverse=True)
        
        for iou, i, j in all_possible_matches:
            if not used_detections[i] and not used_tracks[j]:
                matches.append((i, j))
                used_detections[i] = True
                used_tracks[j] = True
        
        return matches

# ==================== 卡尔曼滤波器实现 ====================
class KalmanFilter2D:
    """2D卡尔曼滤波器(位置和速度)"""
    
    def __init__(self, dt: float = 0.1, 
                 process_noise_std: float = 0.1,
                 measurement_noise_std: float = 1.0):
        """
        初始化2D卡尔曼滤波器
        
        参数:
            dt: 时间步长
            process_noise_std: 过程噪声标准差
            measurement_noise_std: 测量噪声标准差
        """
        self.dt = dt
        
        # 状态向量: [x, y, vx, vy]
        self.state = np.zeros(4)
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 测量矩阵 (只能测量位置)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 过程噪声协方差矩阵
        q = process_noise_std ** 2
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        
        # 测量噪声协方差矩阵
        r = measurement_noise_std ** 2
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        # 状态协方差矩阵
        self.P = np.eye(4) * 100
        
        # 卡尔曼增益
        self.K = np.zeros((4, 2))
        
        # 最后更新时间
        self.last_update_time = time.time()
    
    def predict(self) -> Tuple[float, float]:
        """预测下一时刻的状态"""
        current_time = time.time()
        dt_actual = current_time - self.last_update_time
        
        # 更新状态转移矩阵中的时间步长
        if dt_actual > 0:
            self.F[0, 2] = dt_actual
            self.F[1, 3] = dt_actual
            
            # 更新过程噪声协方差
            q = 0.1 ** 2  # 过程噪声标准差设为0.1
            self.Q = np.array([
                [q*dt_actual**4/4, 0, q*dt_actual**3/2, 0],
                [0, q*dt_actual**4/4, 0, q*dt_actual**3/2],
                [q*dt_actual**3/2, 0, q*dt_actual**2, 0],
                [0, q*dt_actual**3/2, 0, q*dt_actual**2]
            ])
        
        # 状态预测
        self.state = self.F @ self.state
        
        # 协方差预测
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        self.last_update_time = current_time
        return self.state[0], self.state[1]
    
    def update(self, measurement_x: float, measurement_y: float):
        """用测量值更新状态"""
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 测量向量
        z = np.array([measurement_x, measurement_y])
        
        # 状态更新
        y = z - self.H @ self.state
        self.state = self.state + self.K @ y
        
        # 协方差更新
        I = np.eye(4)
        self.P = (I - self.K @ self.H) @ self.P
        
        self.last_update_time = time.time()
    
    def get_state(self) -> Tuple[float, float, float, float]:
        """获取当前状态"""
        return (self.state[0], self.state[1], self.state[2], self.state[3])
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前位置"""
        return (self.state[0], self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取当前速度"""
        return (self.state[2], self.state[3])

# ==================== 机器人跟踪器类 ====================
class RobotTracker:
    """单个机器人的跟踪器（集成卡尔曼滤波）"""
    
    def __init__(self, robot_id: str, initial_x: float, initial_y: float):
        """
        初始化机器人跟踪器
        
        参数:
            robot_id: 机器人ID (如 "R1", "B2")
            initial_x: 初始x坐标
            initial_y: 初始y坐标
        """
        self.robot_id = robot_id
        self.kalman_filter = KalmanFilter2D(dt=0.1)
        
        # 初始化卡尔曼滤波器状态
        self.kalman_filter.state = np.array([initial_x, initial_y, 0, 0])
        
        # 跟踪状态
        self.last_detection_time = time.time()
        self.consecutive_misses = 0
        self.total_detections = 1
        self.is_active = True
        
        # 历史轨迹
        self.trajectory = []
        self.add_to_trajectory(initial_x, initial_y)
        
        # 预测框（用于数据关联）
        self.predicted_box = None
    
    def update(self, x: float, y: float):
        """用新的测量值更新跟踪器"""
        self.kalman_filter.update(x, y)
        self.last_detection_time = time.time()
        self.consecutive_misses = 0
        self.total_detections += 1
        
        # 更新轨迹
        self.add_to_trajectory(x, y)
        
        # 重置预测框
        self.predicted_box = None
    
    def predict(self) -> Tuple[float, float]:
        """预测下一时刻的位置"""
        predicted_x, predicted_y = self.kalman_filter.predict()
        
        # 创建预测框（用于数据关联）
        if self.total_detections > 3:
            # 基于速度估计框的大小
            vx, vy = self.kalman_filter.get_velocity()
            speed = math.sqrt(vx**2 + vy**2)
            box_size = max(20, min(100, 30 + speed * 5))
            self.predicted_box = (predicted_x - box_size/2, 
                                 predicted_y - box_size/2, 
                                 box_size, box_size)
        
        return predicted_x, predicted_y
    
    def add_to_trajectory(self, x: float, y: float):
        """添加点到轨迹历史"""
        self.trajectory.append((x, y, time.time()))
        # 保持轨迹长度不超过50
        if len(self.trajectory) > 50:
            self.trajectory.pop(0)
    
    def get_predicted_box(self) -> Optional[Tuple[float, float, float, float]]:
        """获取预测框（用于数据关联）"""
        return self.predicted_box
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前位置估计"""
        return self.kalman_filter.get_position()
    
    def get_velocity(self) -> Tuple[float, float]:
        """获取当前速度估计"""
        return self.kalman_filter.get_velocity()
    
    def get_state_vector(self) -> Tuple[float, float, float, float]:
        """获取完整状态向量"""
        return self.kalman_filter.get_state()
    
    def update_miss(self):
        """更新未检测到的次数"""
        self.consecutive_misses += 1
        # 如果连续多次未检测到，标记为非活跃
        if self.consecutive_misses > 10:
            self.is_active = False
    
    def is_stale(self, timeout: float = 3.0) -> bool:
        """检查跟踪器是否过期（长时间未更新）"""
        return (time.time() - self.last_detection_time) > timeout

# ==================== 跟踪管理器类 ====================
class TrackingManager:
    """管理所有机器人跟踪器"""
    
    def __init__(self):
        """初始化跟踪管理器"""
        self.trackers: Dict[str, RobotTracker] = {}
        self.max_miss_before_removal = 15
        self.iou_threshold = 0.3
    
    def update(self, detections: List[Tuple[str, Tuple, float]]) -> Dict[str, Tuple[float, float]]:
        """
        用新的检测结果更新所有跟踪器
        
        参数:
            detections: 检测结果列表 [(cls, (x,y,w,h), conf), ...]
            
        返回:
            当前活跃跟踪器的位置字典 {robot_id: (x, y)}
        """
        # 预测所有现有跟踪器的位置
        for tracker in self.trackers.values():
            if tracker.is_active:
                tracker.predict()
        
        # 数据关联：匹配检测到跟踪器
        det_list = []
        for det in detections:
            cls, (x, y, w, h), conf = det
            det_list.append((cls, (x, y, w, h), conf))
        
        track_list = list(self.trackers.values())
        matches = HungarianAlgorithm.match_detections_to_tracks(
            det_list, track_list, self.iou_threshold
        )
        
        # 处理匹配
        matched_det_indices = set()
        matched_track_indices = set()
        
        for det_idx, track_idx in matches:
            det_cls, det_box, det_conf = det_list[det_idx]
            tracker = track_list[track_idx]
            
            # 检查类别是否匹配
            if det_cls == tracker.robot_id:
                x, y, w, h = det_box
                center_x, center_y = x + w/2, y + h/2
                tracker.update(center_x, center_y)
                matched_det_indices.add(det_idx)
                matched_track_indices.add(track_idx)
        
        # 处理未匹配的检测（创建新跟踪器）
        for i, (det_cls, det_box, det_conf) in enumerate(det_list):
            if i not in matched_det_indices:
                x, y, w, h = det_box
                center_x, center_y = x + w/2, y + h/2
                
                # 创建新跟踪器
                if det_cls not in self.trackers:
                    self.trackers[det_cls] = RobotTracker(det_cls, center_x, center_y)
                else:
                    # 如果已有跟踪器但未匹配，重新初始化
                    self.trackers[det_cls] = RobotTracker(det_cls, center_x, center_y)
        
        # 处理未匹配的跟踪器（标记为未检测到）
        tracker_keys = list(self.trackers.keys())
        for i, tracker_key in enumerate(tracker_keys):
            if i not in matched_track_indices:
                tracker = self.trackers[tracker_key]
                tracker.update_miss()
                
                # 移除过期的跟踪器
                if tracker.is_stale() or not tracker.is_active:
                    del self.trackers[tracker_key]
        
        # 返回所有活跃跟踪器的位置
        positions = {}
        for robot_id, tracker in self.trackers.items():
            if tracker.is_active:
                x, y = tracker.get_position()
                positions[robot_id] = (x, y)
        
        return positions
    
    def get_tracker(self, robot_id: str) -> Optional[RobotTracker]:
        """获取指定机器人的跟踪器"""
        return self.trackers.get(robot_id)
    
    def get_all_positions(self) -> Dict[str, Tuple[float, float]]:
        """获取所有活跃跟踪器的位置"""
        positions = {}
        for robot_id, tracker in self.trackers.items():
            if tracker.is_active:
                x, y = tracker.get_position()
                positions[robot_id] = (x, y)
        return positions

# ==================== 配置参数 ====================
# 系统配置
STATE = 'R'  # R:红方，B:蓝方
USART = 1  # 0:关闭串口，1:打开串口
USER_COM = 'COM8'  # 串口号
USER_MODE = 'test'  # 'test':测试模式,'hik':海康相机,'video':USB相机
USER_MAP = 'images/2025map.png'  # 战场地图，2800*1500，左下角坐标原点
USER_IMG_TEST = 'images/test_image.jpg'  # 测试图片
USER_EXPOSURE_TIME = 20000  # 海康相机曝光
USER_GAIN = 16  # 海康相机gain

# 视频录制配置
SAVE_IMG = 0  # 0:关闭视频录制，1:开启视频录制
GAME_DIR = "tete"  # 视频保存的根目录
VIDEO_DIR_MAP = "save_video/" + GAME_DIR + "/map/"
VIDEO_DIR_RAW = "save_video/" + GAME_DIR + "/raw/"
VIDEO_DIR_UI = "save_video/" + GAME_DIR + "/ui/"

# ==================== 初始化全局变量 ====================
# 加载标定数据和掩码
if STATE == 'R':
    try:
        loaded_arrays = np.load('arrays_test_red.npy')
        mask_image = cv2.imread("images/2025map_mask.png")
    except:
        print("警告: 无法加载红方标定数据，使用默认值")
        loaded_arrays = np.array([np.eye(3), np.eye(3), np.eye(3)])
        mask_image = np.zeros((1500, 2800, 3), dtype=np.uint8)
else:
    try:
        loaded_arrays = np.load('arrays_test_blue.npy')
        mask_image = cv2.imread("images/2025map_mask.png")
    except:
        print("警告: 无法加载蓝方标定数据，使用默认值")
        loaded_arrays = np.array([np.eye(3), np.eye(3), np.eye(3)])
        mask_image = np.zeros((1500, 2800, 3), dtype=np.uint8)

# 导入战场每个高度的不同仿射变化矩阵
M_HEIGHT_R = loaded_arrays[1]  # R型高地
M_HEIGHT_G = loaded_arrays[2]  # 环形高地
M_GROUND = loaded_arrays[0]  # 地面层、公路层

# 确定地图画面像素，保证不会溢出
HEIGHT, WIDTH = mask_image.shape[:2]
HEIGHT -= 1
WIDTH -= 1

# 初始化UI和状态变量
information_ui = np.zeros((500, 420, 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()
double_vulnerability_chance = -1  # 双倍易伤机会数
opponent_double_vulnerability = -1  # 是否正在触发双倍易伤
target = -1  # 飞镖当前瞄准目标
chances_flag = 1  # 双倍易伤触发标志位
progress_list = [-1, -1, -1, -1, -1, -1]  # 标记进度列表

# 加载战场地图
try:
    map_backup = cv2.imread(USER_MAP)
    if map_backup is None:
        raise FileNotFoundError
except:
    print("警告: 无法加载地图，创建空白地图")
    map_backup = np.zeros((1500, 2800, 3), dtype=np.uint8)
    cv2.putText(map_backup, "MAP NOT FOUND", (1000, 750), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

map_display = map_backup.copy()

# 初始化盲区预测列表
guess_list = {
    "B1": True, "B2": True, "B3": True, "B4": True, "B5": True, "B6": True, "B7": True,
    "R1": True, "R2": True, "R3": True, "R4": True, "R5": True, "R6": True, "R7": True
}

# 机器人名字对应ID
mapping_table = {
    "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5, "R6": 6, "R7": 7,
    "B1": 101, "B2": 102, "B3": 103, "B4": 104, "B5": 105, "B6": 106, "B7": 107
}

# 盲区预测点位
guess_table = {
    "R1": [(1000, 400), (960, 1000), (1123, 1195), (800, 1225), (946, 1341), (457, 1232)],
    "R2": [(200, 100), (900, 900), (900, 600), (1335, 821), (1469, 687)],
    "R3": [(998, 1059), (1186, 1266), (1663, 246)],
    "R4": [(998, 1059), (1186, 1266), (1663, 246)],
    "R7": [(386, 812), (1356, 1093), (1179, 858)],
    
    "B1": [(1821, 1092), (1851, 513), (1754, 403), (2050, 347), (1800, 200)],
    "B2": [(2600, 1400), (1900, 636), (1900, 878), (1500, 750), (1410, 654)],
    "B3": [(1814, 475), (784, 1372), (1646, 270)],
    "B4": [(1814, 475), (784, 1372), (1646, 270)],
    "B7": [(1979, 652)],
}

# 盲区预测参数
D_FACTOR = 0.01
COS_FACTOR = 0.003

# ==================== 相机获取函数 ====================
camera_image = None

def hik_camera_get():
    """海康相机图像获取线程"""
    global camera_image
    try:
        # 导入海康相机模块
        from hik_camera import (
            start_grab_and_get_data_size, 
            set_Value, 
            get_Value, 
            image_control
        )
        
        # 根据平台导入相应的相机控制类
        if sys.platform.startswith("win"):
            from MvImport.MvCameraControl_class import (
                MvCamera, MV_CC_DEVICE_INFO_LIST, MV_GIGE_DEVICE, MV_USB_DEVICE,
                MV_ACCESS_Exclusive, MVCC_INTVALUE_EX, MV_FRAME_OUT_INFO_EX,
                memset, byref, sizeof, c_ubyte, cast, POINTER, MV_CC_DEVICE_INFO
            )
        else:
            from MvImport_Linux.MvCameraControl_class import (
                MvCamera, MV_CC_DEVICE_INFO_LIST, MV_GIGE_DEVICE, MV_USB_DEVICE,
                MV_ACCESS_Exclusive, MVCC_INTVALUE_EX, MV_FRAME_OUT_INFO_EX,
                memset, byref, sizeof, c_ubyte, cast, POINTER, MV_CC_DEVICE_INFO
            )
    except ImportError as e:
        print(f"无法导入海康相机模块: {e}")
        return
    
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    
    # 枚举设备
    while True:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"枚举设备失败! ret[0x{ret:x}]")
            time.sleep(1)
            continue
            
        if deviceList.nDeviceNum == 0:
            print("未找到设备!")
            time.sleep(1)
            continue
        else:
            print(f"找到 {deviceList.nDeviceNum} 个设备!")
            break
    
    # 选择第一个设备
    nConnectionNum = 0
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[nConnectionNum], POINTER(MV_CC_DEVICE_INFO)).contents
    
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"创建句柄失败! ret[0x{ret:x}]")
        return
    
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"打开设备失败! ret[0x{ret:x}]")
        return
    
    print("相机参数:", 
          get_Value(cam, param_type="float_value", node_name="ExposureTime"),
          get_Value(cam, param_type="float_value", node_name="Gain"),
          get_Value(cam, param_type="enum_value", node_name="TriggerMode"),
          get_Value(cam, param_type="float_value", node_name="AcquisitionFrameRate"))
    
    set_Value(cam, param_type="float_value", node_name="ExposureTime", node_value=USER_EXPOSURE_TIME)
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=USER_GAIN)
    
    start_grab_and_get_data_size(cam)
    
    stParam = MVCC_INTVALUE_EX()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print(f"获取数据包大小失败! ret[0x{ret:x}]")
        return
    
    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(pData)
            camera_image = image_control(data=image, stFrameInfo=stFrameInfo)
        else:
            print(f"无数据[0x{ret:x}]")

def video_capture_get():
    """USB相机图像获取线程"""
    global camera_image
    cam = cv2.VideoCapture(0)  # 默认摄像头
    
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)  # 尝试第二个摄像头
    
    while True:
        ret, img = cam.read()
        if ret:
            camera_image = img
            time.sleep(0.016)  # 约60fps
        else:
            print("无法从摄像头读取图像")
            time.sleep(0.1)

# ==================== 串口通信函数 ====================
def get_low_order_bit_list(received_data):
    """从接收的数据中提取低位比特列表"""
    if isinstance(received_data, bytes):
        byte_value = received_data[0]
    else:
        byte_value = received_data
    
    bit_list = [(byte_value >> i) & 1 for i in range(8)]
    bit_list.insert(4, 0)
    bit_list = [x * 120 for x in bit_list]
    for _ in range(3):
        bit_list.pop()
    
    return bit_list

def ser_send():
    """串口发送线程"""
    if not SERIAL_MODULE_AVAILABLE:
        print("串口模块不可用，跳过串口发送线程")
        return
    
    seq = 0
    global chances_flag
    
    # 发送地图坐标
    send_map = {
        "R1": (0, 0), "R2": (0, 0), "R3": (0, 0), "R4": (0, 0), "R5": (0, 0), "R6": (0, 0), "R7": (0, 0),
        "B1": (0, 0), "B2": (0, 0), "B3": (0, 0), "B4": (0, 0), "B5": (0, 0), "B6": (0, 0), "B7": (0, 0)
    }
    
    while True:
        try:
            # 获取跟踪器的位置
            all_tracked_positions = tracking_manager.get_all_positions()
            
            # 根据阵营处理坐标
            for robot_id, (x, y) in all_tracked_positions.items():
                if STATE == 'R':
                    # 红方视角：蓝方为敌方
                    if robot_id.startswith('B'):
                        filtered_xyz = (2800 - y, x)  # 坐标转换
                        ser_x = int(filtered_xyz[0]) * 10 / 10
                        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
                        send_map[robot_id] = (ser_x, ser_y)
                else:
                    # 蓝方视角：红方为敌方
                    if robot_id.startswith('R'):
                        filtered_xyz = (y, 1500 - x)  # 坐标转换
                        ser_x = int(filtered_xyz[0]) * 10 / 10
                        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
                        send_map[robot_id] = (ser_x, ser_y)
            
            # 发送数据
            ser_data = build_data_radar_all(send_map, STATE)
            packet, seq = build_send_packet(ser_data, seq, [0x03, 0x05])
            ser1.write(packet)
            time.sleep(0.2)
            
        except Exception as e:
            print(f'串口发送错误: {e}')
            time.sleep(0.5)

def ser_receive():
    """串口接收线程"""
    if not SERIAL_MODULE_AVAILABLE:
        print("串口模块不可用，跳过串口接收线程")
        return
    
    global progress_list
    global double_vulnerability_chance
    global opponent_double_vulnerability
    global target
    
    progress_cmd_id = [0x02, 0x0C]  # 雷达标记进度
    vulnerability_cmd_id = [0x02, 0x0E]  # 双倍易伤
    target_cmd_id = [0x01, 0x05]  # 飞镖目标
    
    buffer = b''
    
    while True:
        try:
            received_data = ser1.read_all()
            buffer += received_data
            sof_index = buffer.find(b'\xA5')
            
            while sof_index != -1:
                if len(buffer) >= sof_index + 5:
                    packet_data = buffer[sof_index:]
                    next_sof_index = packet_data.find(b'\xA5', 1)
                    
                    if next_sof_index != -1:
                        packet_data = packet_data[:next_sof_index]
                    
                    # 解析数据包
                    progress_result = receive_packet(packet_data, progress_cmd_id, info=False)
                    vulnerability_result = receive_packet(packet_data, vulnerability_cmd_id, info=False)
                    target_result = receive_packet(packet_data, target_cmd_id, info=False)
                    
                    # 更新数据
                    if progress_result is not None:
                        received_cmd_id1, received_data1, received_seq1 = progress_result
                        progress_list = get_low_order_bit_list(received_data1)
                    
                    if vulnerability_result is not None:
                        received_cmd_id2, received_data2, received_seq2 = vulnerability_result
                        received_data2 = list(received_data2)[0]
                        double_vulnerability_chance, opponent_double_vulnerability = Radar_decision(received_data2)
                    
                    if target_result is not None:
                        received_cmd_id3, received_data3, received_seq3 = target_result
                        target = (list(received_data3)[1] & 0b11000000) >> 6
                    
                    buffer = buffer[sof_index + len(packet_data):]
                    sof_index = buffer.find(b'\xA5')
                else:
                    break
            
        except Exception as e:
            print(f'串口接收错误: {e}')
        
        time.sleep(0.1)

# ==================== 坐标转换函数 ====================
def image_to_map_coordinates(x: float, y: float, cls: str) -> Tuple[float, float, str]:
    """
    将图像坐标转换为地图坐标
    
    参数:
        x, y: 图像坐标（装甲板中心下沿）
        cls: 机器人类别
        
    返回:
        (x_map, y_map, height_layer)
    """
    # 原图中装甲板的中心下沿作为待仿射变化的点
    camera_point = np.array([[[min(x, WIDTH), min(y, HEIGHT)]]], dtype=np.float32)
    
    # 低到高依次仿射变化
    # 先套用地面层仿射变化矩阵
    mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_GROUND)
    
    # 限制转换后的点在地图范围内
    x_c = max(int(mapped_point[0][0][0]), 0)
    y_c = max(int(mapped_point[0][0][1]), 0)
    x_c = min(x_c, WIDTH)
    y_c = min(y_c, HEIGHT)
    
    # 通过掩码图像判断高度层
    color = mask_image[y_c, x_c]
    
    if color[0] == color[1] == color[2] == 0:
        # 黑色：地面层
        X_M = x_c
        Y_M = y_c
        height_layer = "ground"
    elif color[1] > color[2] and color[1] > color[0]:
        # 绿色：R型高地
        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_HEIGHT_R)
        x_c = max(int(mapped_point[0][0][0]), 0)
        y_c = max(int(mapped_point[0][0][1]), 0)
        x_c = min(x_c, WIDTH)
        y_c = min(y_c, HEIGHT)
        X_M = x_c
        Y_M = y_c
        height_layer = "height_r"
    elif color[0] > color[2] and color[0] > color[1]:
        # 蓝色：环形高地
        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_HEIGHT_G)
        x_c = max(int(mapped_point[0][0][0]), 0)
        y_c = max(int(mapped_point[0][0][1]), 0)
        x_c = min(x_c, WIDTH)
        y_c = min(y_c, HEIGHT)
        X_M = x_c
        Y_M = y_c
        height_layer = "height_g"
    else:
        # 默认使用R型高地
        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_HEIGHT_R)
        x_c = max(int(mapped_point[0][0][0]), 0)
        y_c = max(int(mapped_point[0][0][1]), 0)
        x_c = min(x_c, WIDTH)
        y_c = min(y_c, HEIGHT)
        X_M = x_c
        Y_M = y_c
        height_layer = "default_r"
    
    return X_M, Y_M, height_layer

# ==================== 初始化系统组件 ====================
# 初始化跟踪管理器
tracking_manager = TrackingManager()

# 加载YOLOv11模型
try:
    weights_path = 'models/car.engine' if os.path.exists('models/car.engine') else 'models/car.onnx'
    weights_path_next = 'models/armor.engine' if os.path.exists('models/armor.engine') else 'models/armor.onnx'
    
    detector = YOLOv11Detector(
        weights_path=weights_path,
        img_size=640,
        conf_thres=0.1,
        iou_thres=0.5,
        max_det=14,
        data='yaml/car.yaml',
        ui=True
    )
    
    detector_next = YOLOv11Detector(
        weights_path=weights_path_next,
        img_size=640,
        conf_thres=0.4,
        iou_thres=0.2,
        max_det=1,
        data='yaml/armor.yaml',
        ui=True
    )
    print("YOLOv11模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
    print("将使用模拟检测器")
    detector = YOLOv11Detector('')
    detector_next = YOLOv11Detector('')

# 初始化串口
ser1 = None
if USART and SERIAL_MODULE_AVAILABLE:
    try:
        ser1 = serial.Serial(USER_COM, 115200, timeout=1)
        print(f"串口 {USER_COM} 打开成功")
        
        # 启动串口线程
        thread_receive = threading.Thread(target=ser_receive, daemon=True)
        thread_receive.start()
        thread_send = threading.Thread(target=ser_send, daemon=True)
        thread_send.start()
    except Exception as e:
        print(f"串口初始化失败: {e}")
        ser1 = None

# 初始化相机
camera_mode = USER_MODE
camera_image = None

if camera_mode == 'test':
    if os.path.exists(USER_IMG_TEST):
        ext = os.path.splitext(USER_IMG_TEST)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            camera_image = cv2.imread(USER_IMG_TEST)
            print(f"加载测试图像: {USER_IMG_TEST}")
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_capture = cv2.VideoCapture(USER_IMG_TEST)
            ret, camera_image = video_capture.read()
            if ret:
                print(f"加载测试视频: {USER_IMG_TEST}")
            else:
                print(f"无法读取测试视频: {USER_IMG_TEST}")
        else:
            print(f"不支持的测试文件格式: {USER_IMG_TEST}")
    else:
        print(f"测试文件不存在: {USER_IMG_TEST}")

elif camera_mode in ['hik', 'hik_test']:
    thread_camera = threading.Thread(target=hik_camera_get, daemon=True)
    thread_camera.start()
    print("启动海康相机线程")

elif camera_mode == 'video':
    thread_camera = threading.Thread(target=video_capture_get, daemon=True)
    thread_camera.start()
    print("启动USB相机线程")

# 等待图像
start_wait = time.time()
while camera_image is None:
    print("等待图像...")
    time.sleep(0.5)
    if time.time() - start_wait > 10:
        print("图像等待超时，使用测试图像")
        # 创建测试图像
        camera_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(camera_image, "NO CAMERA FEED", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        break

print("图像准备就绪")

# 初始化视频录制
video_writer_map = None
video_writer_raw = None
video_writer_ui = None

if SAVE_IMG:
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10
        
        os.makedirs(VIDEO_DIR_MAP, exist_ok=True)
        os.makedirs(VIDEO_DIR_RAW, exist_ok=True)
        os.makedirs(VIDEO_DIR_UI, exist_ok=True)
        
        video_path1 = os.path.join(VIDEO_DIR_MAP, f"map_{timestamp}.avi")
        video_path2 = os.path.join(VIDEO_DIR_RAW, f"screen_{timestamp}.avi")
        video_path3 = os.path.join(VIDEO_DIR_UI, f"screen_{timestamp}.avi")
        
        video_writer_map = cv2.VideoWriter(video_path1, fourcc, fps, (600, 320))
        video_writer_raw = cv2.VideoWriter(video_path2, fourcc, fps, (1300, 900))
        video_writer_ui = cv2.VideoWriter(video_path3, fourcc, fps, (1300, 900))
        
        print("视频录制已初始化")
    except Exception as e:
        print(f"视频录制初始化失败: {e}")

# ==================== 主循环 ====================
print("=" * 50)
print("RoboMaster 雷达视觉系统启动")
print(f"阵营: {STATE}, 相机模式: {camera_mode}")
print("=" * 50)

frame_count = 0
fps_history = deque(maxlen=30)

while True:
    frame_count += 1
    start_time = time.time()
    
    # 刷新UI和地图
    information_ui_show = information_ui.copy()
    map_display = map_backup.copy()
    
    # 获取图像
    if camera_mode == 'test' and 'video_capture' in locals():
        ret, camera_image = video_capture.read()
        if not ret:
            print("测试视频结束")
            break
    
    if camera_image is None:
        print("无图像数据")
        time.sleep(0.1)
        continue
    
    img0 = camera_image.copy()
    img_height, img_width = img0.shape[:2]
    
    # 保存原始图像
    if SAVE_IMG and video_writer_raw is not None:
        resized_raw = cv2.resize(img0, (1300, 900))
        video_writer_raw.write(resized_raw)
    
    # 第一层神经网络识别（车体检测）
    detections_primary = []
    try:
        result0 = detector.predict(img0)
        
        for detection in result0:
            cls, xywh, conf = detection
            if cls == 'car':
                left, top, w, h = xywh
                left, top, w, h = int(left), int(top), int(w), int(h)
                
                # 存储检测结果
                detections_primary.append((cls, (left, top, w, h), conf))
                
                # ROI出机器人区域
                cropped = camera_image[top:top + h, left:left + w]
                cropped_img = np.ascontiguousarray(cropped)
                
                # 第二层神经网络识别（装甲板检测）
                result_n = detector_next.predict(cropped_img)
                
                if result_n:
                    # 叠加第二次检测结果到原图的对应位置
                    img0[top:top + h, left:left + w] = cropped_img
                    
                    for detection1 in result_n:
                        cls_armor, xywh_armor, conf_armor = detection1
                        if cls_armor and cls_armor in ['R1', 'R2', 'R3', 'R4', 'R5', 'R7', 
                                                      'B1', 'B2', 'B3', 'B4', 'B5', 'B7']:
                            x_armor, y_armor, w_armor, h_armor = xywh_armor
                            x_armor = x_armor + left
                            y_armor = y_armor + top
                            
                            # 坐标转换：图像坐标 -> 地图坐标
                            x_map, y_map, height_layer = image_to_map_coordinates(
                                x_armor + w_armor/2, 
                                y_armor + h_armor,
                                cls_armor
                            )
                            
                            # 添加到检测结果（用于跟踪）
                            detections_primary.append((
                                cls_armor, 
                                (x_map, y_map, 30, 30),  # 简化框
                                conf_armor
                            ))
    except Exception as e:
        print(f"检测错误: {e}")
    
    # 使用跟踪管理器更新跟踪器
    tracked_positions = tracking_manager.update(detections_primary)
    
    # 在地图上绘制敌方机器人
    for robot_id, (x, y) in tracked_positions.items():
        # 检查是否为敌方机器人
        if (STATE == 'R' and robot_id.startswith('B')) or \
           (STATE == 'B' and robot_id.startswith('R')):
            
            # 设置颜色
            if robot_id.startswith('R'):
                color_m = (0, 0, 255)  # 红色
            else:
                color_m = (255, 0, 0)  # 蓝色
            
            # 坐标转换到地图显示坐标
            if STATE == 'R':
                # 红方视角
                display_x = 2800 - y
                display_y = x
            else:
                # 蓝方视角
                display_x = y
                display_y = 1500 - x
            
            # 绘制机器人位置
            cv2.circle(map_display, (int(display_x), int(display_y)), 15, color_m, -1)
            
            # 绘制机器人ID
            cv2.putText(map_display, robot_id,
                       (int(display_x) - 15, int(display_y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 绘制坐标
            coord_text = f"({int(x)}, {int(y)})"
            cv2.putText(map_display, coord_text,
                       (int(display_x) - 40, int(display_y) + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 获取跟踪器状态
            tracker = tracking_manager.get_tracker(robot_id)
            if tracker:
                vx, vy = tracker.get_velocity()
                speed = math.sqrt(vx**2 + vy**2)
                
                # 绘制速度矢量
                if speed > 0.1:
                    end_x = int(display_x + vx * 10)
                    end_y = int(display_y + vy * 10)
                    cv2.arrowedLine(map_display, 
                                   (int(display_x), int(display_y)),
                                   (end_x, end_y),
                                   (0, 255, 0), 2)
    
    # 绘制UI
    height_light = draw_information_ui(progress_list, STATE, information_ui_show)
    
    # 显示系统状态
    cv2.putText(information_ui_show, f"Trackers: {len(tracking_manager.trackers)}",
                (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(information_ui_show, f"FPS: {1/fps_history[-1]:.1f}" if fps_history else "FPS: -",
                (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(information_ui_show, f"Frame: {frame_count}",
                (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 显示图像
    map_show = cv2.resize(map_display, (600, 320))
    img0_show = cv2.resize(img0, (1300, 900))
    
    cv2.imshow('information_ui', information_ui_show)
    cv2.imshow('map', map_show)
    cv2.imshow('img', img0_show)
    
    # 保存视频
    if SAVE_IMG:
        if video_writer_map is not None:
            video_writer_map.write(map_show)
        if video_writer_ui is not None:
            video_writer_ui.write(img0_show)
    
    # 计算FPS
    end_time = time.time()
    frame_time = end_time - start_time
    fps_history.append(frame_time)
    
    # 显示FPS
    avg_fps = 1.0 / (sum(fps_history) / len(fps_history)) if fps_history else 0
    fps_text = f"FPS: {avg_fps:.1f} | Trackers: {len(tracked_positions)}"
    cv2.putText(img0_show, fps_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 显示算法状态
    algo_text = "KF+HA"  # 卡尔曼滤波+匈牙利算法
    cv2.putText(img0_show, algo_text, (img0_show.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # 按键处理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # 'q' 或 ESC
        print("用户退出程序")
        break
    elif key == ord('s'):  # 保存快照
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"snapshot_{timestamp}.jpg", img0_show)
        print(f"已保存快照: snapshot_{timestamp}.jpg")
    elif key == ord('r'):  # 重置跟踪器
        tracking_manager.trackers.clear()
        print("跟踪器已重置")
    elif key == ord('p'):  # 暂停
        print("程序暂停，按任意键继续...")
        cv2.waitKey(0)
    elif key == ord('d'):  # 调试信息
        print(f"活跃跟踪器: {len(tracking_manager.trackers)}")
        for robot_id, tracker in tracking_manager.trackers.items():
            x, y = tracker.get_position()
            vx, vy = tracker.get_velocity()
            print(f"  {robot_id}: ({x:.1f}, {y:.1f}) v=({vx:.1f}, {vy:.1f})")

# ==================== 清理资源 ====================
print("正在关闭程序...")

cv2.destroyAllWindows()

if SAVE_IMG:
    if video_writer_map is not None:
        video_writer_map.release()
        print("地图视频已保存")
    if video_writer_raw is not None:
        video_writer_raw.release()
        print("原始视频已保存")
    if video_writer_ui is not None:
        video_writer_ui.release()
        print("UI视频已保存")

if ser1 is not None:
    ser1.close()
    print("串口已关闭")

print("程序退出完成")