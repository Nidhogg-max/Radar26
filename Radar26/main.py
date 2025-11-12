import math
import threading
import time
from collections import deque
import datetime

import serial 

from detect_function_yolov11 import YOLOv11Detector
from information_ui import draw_information_ui

import sys
import os
import cv2
import numpy as np
from detect_function_yolov11 import YOLOv11Detector
from RM_serial_py.ser_api import build_send_packet, receive_packet, Radar_decision, \
    build_data_decision, build_data_radar_all, build_data_sentry

# =============================================================================
# 配置参数
# =============================================================================

# 全局配置
state = 'B'  # R:红方/B:蓝方
USART = True  # 是否启用串口
user_com = 'COM19'  # 串口号
user_mode = 'test'  # 'test':测试模式,'hik':海康相机,'video':USB相机
user_map = 'images/map.jpg'  # 战场地图
user_img_test = 'images/test_image.jpg'  # 测试图片

# 相机参数
user_ExposureTime = 16000  # 海康相机曝光
user_Gain = 17.9  # 海康相机gain

# =============================================================================
# 相机内参和畸变系数（需要根据实际相机标定结果修改）
# =============================================================================

# 相机内参矩阵 [fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 需要替换为实际相机标定结果
camera_matrix = np.array([
    [1000.0, 0, 640],    # [fx, 0, cx]  需要替换为实际值
    [0, 1000.0, 360],    # [0, fy, cy]  需要替换为实际值  
    [0, 0, 1]            # 内参矩阵最后一行
], dtype=np.float32)

# 畸变系数 [k1, k2, p1, p2, k3] 
# 需要替换为实际相机标定结果
dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 需要替换为实际值

# =============================================================================
# 装甲板3D尺寸定义（单位：米）
# =============================================================================

# 假设装甲板是矩形，定义其在世界坐标系中的3D坐标
# 以装甲板中心为原点，平面为XY平面
armor_width = 0.13   # 装甲板宽度（米） 需要根据实际尺寸调整
armor_height = 0.055 # 装甲板高度（米） 需要根据实际尺寸调整

# 装甲板在世界坐标系中的3D点（装甲板中心为原点）
armor_3d_points = np.array([
    [-armor_width/2, -armor_height/2, 0],  # 左上角
    [armor_width/2, -armor_height/2, 0],   # 右上角  
    [armor_width/2, armor_height/2, 0],    # 右下角
    [-armor_width/2, armor_height/2, 0]    # 左下角
], dtype=np.float32)

# =============================================================================
# 初始化配置和数据结构
# =============================================================================

# 加载标定矩阵和地图
if state == 'R':
    loaded_arrays = np.load('arrays_test_red.npy')
    mask_image = cv2.imread("images/map_mask.jpg")
else:
    loaded_arrays = np.load('arrays_test_blue.npy')
    mask_image = cv2.imread("images/map_mask.jpg")

# 导入战场每个高度的不同仿射变化矩阵
M_height_r = loaded_arrays[1]  # R型高地
M_height_g = loaded_arrays[2]  # 环形高地
M_ground = loaded_arrays[0]    # 地面层、公路层

# 确定地图画面像素，保证不会溢出
height, width = mask_image.shape[:2]
height -= 1
width -= 1

# 初始化战场信息UI
information_ui = np.zeros((500, 420, 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()
double_vulnerability_chance = -1  # 双倍易伤机会数
opponent_double_vulnerability = -1  # 是否正在触发双倍易伤
target = -1  # 飞镖当前瞄准目标
chances_flag = 1  # 双倍易伤触发标志位
progress_list = [-1, -1, -1, -1, -1, -1]  # 标记进度列表

# 加载战场地图
map_backup = cv2.imread(user_map)
map = map_backup.copy()

# =============================================================================
# 盲区预测系统
# =============================================================================

# 初始化盲区预测列表
guess_list = {
    "B1": True, "B2": True, "B3": True, "B4": True, "B5": True, "B6": True, "B7": True,
    "R1": True, "R2": True, "R3": True, "R4": True, "R5": True, "R6": True, "R7": True
}

# 上次盲区预测时的标记进度
guess_value = {"B1": 0, "B2": 0, "B7": 0, "R1": 0, "R2": 0, "R7": 0}

# 当前标记进度（用于判断是否预测正确）
guess_value_now = {"B1": 0, "B2": 0, "B7": 0, "R1": 0, "R2": 0, "R7": 0}

# 机器人名字对应ID
mapping_table = {
    "R1": 1, "R2": 2, "R3": 3, "R4": 4, "R5": 5, "R6": 6, "R7": 7,
    "B1": 101, "B2": 102, "B3": 103, "B4": 104, "B5": 105, "B6": 106, "B7": 107
}

# 盲区预测点位
guess_table = {
    "R1": [(1100, 1400), (900, 1400)],
    "R2": [(870, 1100), (1340, 680)],
    "R7": [(560, 630), (560, 870)],
    "B1": [(1700, 100), (1900, 100)],
    "B2": [(1930, 400), (1460, 820)],
    "B7": [(2240, 870), (2240, 603)],
}

class Predict:
    """盲区预测类"""
    global guess_table

    def __init__(self):
        self.trajectory = []
        self.flag = False

    def add_point(self, point):
        self.trajectory.append(point)

    def clear_point(self):
        if len(self.trajectory) > 105:
            del self.trajectory[:100]

    def predict_point(self, guess_points):
        if len(self.trajectory) < 2:
            return sorted(guess_points, key=lambda p: math.sqrt(p[0] ** 2 + p[1] ** 2))

        if not guess_points:
            return []

        # 计算速度向量
        last_pos = self.trajectory[-1]
        prev_pos = self.trajectory[-2]
        v_vector = (last_pos[0] - prev_pos[0], last_pos[1] - prev_pos[1])

        scores = []

        for point in guess_points:
            # 计算到固定点的向量
            d_vector = (point[0] - last_pos[0], point[1] - last_pos[1])

            # 计算余弦相似度
            dot_product = v_vector[0] * d_vector[0] + v_vector[1] * d_vector[1]
            v_norm = math.sqrt(v_vector[0] ** 2 + v_vector[1] ** 2)
            d_norm = math.sqrt(d_vector[0] ** 2 + d_vector[1] ** 2)
            cos_sim = dot_product / (v_norm * d_norm + 1e-8)  # 避免除零

            # 计算欧式距离
            distance = d_norm
            d_score = math.exp(-distance * 0.01)  # d_factor = 0.01

            # 分数值确定优先级
            score = 0.003 * cos_sim + (1 - 0.003) * d_score  # cos_factor = 0.003
            scores.append((point, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scores]

    def get_points(self, name):
        if self.flag:
            guess_table[name] = self.predict_point(guess_table.get(name))
            self.flag = False

# 初始化盲区预测器
guess_predict = {
    "B1": Predict(), "B2": Predict(), "B7": Predict(),
    "R1": Predict(), "R2": Predict(), "R7": Predict()
}

# =============================================================================
# 定位算法系统 融合仿射变换、PnP和卡尔曼滤波
# =============================================================================

class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器 - 针对小车急刹车、急转弯等非线性运动
    使用匀速转向模型(CTRV)来更好地处理非线性运动
    """
    def __init__(self, state_dim=4, measure_dim=2):
        # 状态向量: [x, y, v, theta]  位置(x,y), 速度, 航向角
        self.state_dim = state_dim
        self.measure_dim = measure_dim
        
        # 状态向量
        self.state = np.zeros((state_dim, 1), dtype=np.float32)
        
        # 状态协方差矩阵
        self.P = np.eye(state_dim, dtype=np.float32) * 100
        
        # 过程噪声协方差
        self.Q = np.eye(state_dim, dtype=np.float32) * 0.1
        
        # 测量噪声协方差
        self.R = np.eye(measure_dim, dtype=np.float32) * 1.0
        
        # 测量矩阵
        self.H = np.zeros((measure_dim, state_dim), dtype=np.float32)
        self.H[0, 0] = 1  # 测量x
        self.H[1, 1] = 1  # 测量y
        
        self.last_update_time = time.time()
        
    def predict(self, dt=None):
        """预测步骤  使用CTRV模型"""
        if dt is None:
            current_time = time.time()
            dt = current_time - self.last_update_time
            self.last_update_time = current_time
        
        x, y, v, theta = self.state.flatten()
        
        # CTRV模型的状态转移
        if abs(v) < 0.1:  # 速度很小时，使用线性模型
            F = np.eye(self.state_dim, dtype=np.float32)
            F[0, 2] = dt
            F[1, 2] = dt
            self.state = F @ self.state
        else:
            # 非线性CTRV模型
            new_x = x + (v / theta) * (np.sin(theta * dt + theta) - np.sin(theta))
            new_y = y + (v / theta) * (np.cos(theta) - np.cos(theta * dt + theta))
            new_v = v  # 假设速度不变
            new_theta = theta  # 假设航向角不变
            
            self.state = np.array([[new_x], [new_y], [new_v], [new_theta]], dtype=np.float32)
        
        # 预测协方差
        self.P = self.P + self.Q
        
        return self.state
    
    def update(self, measurement):
        # 测量残差
        y = measurement.reshape(-1, 1) - self.H @ self.state
        
        # 卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.state = self.state + K @ y
        
        # 更新协方差
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state
    
    def get_position(self):
        """获取当前位置估计"""
        return self.state[0, 0], self.state[1, 0]

class PnPLocalizer:
    """
    PnP定位器  使用PnP算法进行精确3D定位
    """
    def __init__(self, camera_matrix, dist_coeffs, armor_3d_points):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.armor_3d_points = armor_3d_points
        
    def solve_pnp(self, bbox_2d_points):
        """
        使用PnP算法求解相机位姿
        
        Args:
            bbox_2d_points: 边界框的四个角点坐标 [左上, 右上, 右下, 左下]
            
        Returns:
            success: 是否成功求解
            position: 3D位置坐标 (x, y, z)
        """
        try:
            # 确保输入数据格式正确
            if len(bbox_2d_points) != 4:
                return False, (0, 0, 0)
                
            # 转换为numpy数组
            image_points = np.array(bbox_2d_points, dtype=np.float32)
            
            # 使用EPnP算法求解PnP问题
            success, rvec, tvec = cv2.solvePnP(
                self.armor_3d_points, 
                image_points, 
                self.camera_matrix, 
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_EPNP  # EPnP算法对噪声更鲁棒
            )
            
            if success:
                # tvec就是装甲板在相机坐标系中的3D位置
                x, y, z = tvec.flatten()
                return True, (x, y, z)
            else:
                return False, (0, 0, 0)
                
        except Exception as e:
            print(f"PnP求解错误: {e}")
            return False, (0, 0, 0)
    
    def project_to_map(self, camera_position, map_width=2800, map_height=1500):
        """
        将相机坐标系中的3D位置投影到2D地图
        
        Args:
            camera_position: 相机坐标系中的3D位置 (x, y, z)
            map_width, map_height: 地图尺寸
            
        Returns:
            map_x, map_y: 地图坐标系中的2D坐标
        """
        x, y, z = camera_position
        
        # 假设相机坐标系: X向右, Y向下, Z向前
        # 地图坐标系: X向右, Y向上
        # 转换: 地图X = 相机X, 地图Y = 地图高度 - 相机Z
        
        map_x = int(x * 100 + map_width / 2)  # 缩放并居中
        map_y = int(map_height - z * 100)     # 缩放并翻转Y轴
        
        # 限制在地图范围内
        map_x = max(0, min(map_x, map_width - 1))
        map_y = max(0, min(map_y, map_height - 1))
        
        return map_x, map_y

class MultiLevelLocalization:
    """
    多级定位系统  融合仿射变换、PnP和卡尔曼滤波
    定位流程:
    1. 仿射变换: 快速粗略定位，确定大致区域和高度层
    2. PnP算法: 精确3D定位,获得准确的世界坐标
    3. 卡尔曼滤波: 运动预测和状态平滑，处理急刹车急转弯
    """
    def __init__(self, camera_matrix, dist_coeffs, armor_3d_points, 
                 M_ground, M_height_r, M_height_g, mask_image):
        # 初始化各组件
        self.pnp_localizer = PnPLocalizer(camera_matrix, dist_coeffs, armor_3d_points)
        self.affine_matrices = {
            'ground': M_ground,
            'height_r': M_height_r, 
            'height_g': M_height_g
        }
        self.mask_image = mask_image
        
        # 为每个机器人维护一个EKF
        self.filters = {}
        
        # 地图尺寸
        self.map_height, self.map_width = mask_image.shape[:2]
        self.map_height -= 1
        self.map_width -= 1
        
    def affine_transform_localization(self, camera_point):
        """
        第一阶段: 仿射变换粗略定位
        确定目标所在的高度层和大致区域
        """
        height_level = "ground"  # 默认地面层
        
        # 依次尝试不同高度的仿射变换矩阵
        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), self.affine_matrices['ground'])
        x_c = max(int(mapped_point[0][0][0]), 0)
        y_c = max(int(mapped_point[0][0][1]), 0)
        x_c = min(x_c, self.map_width)
        y_c = min(y_c, self.map_height)
        
        color = self.mask_image[y_c, x_c]
        
        if color[0] == color[1] == color[2] == 0:
            # 地面层
            height_level = "ground"
            X_M, Y_M = x_c, y_c
        else:
            # R型高地
            mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), self.affine_matrices['height_r'])
            x_c = max(int(mapped_point[0][0][0]), 0)
            y_c = max(int(mapped_point[0][0][1]), 0)
            x_c = min(x_c, self.map_width)
            y_c = min(y_c, self.map_height)
            
            color = self.mask_image[y_c, x_c]
            if color[1] > color[2] and color[1] > color[0]:
                height_level = "height_r"
                X_M, Y_M = x_c, y_c
            else:
                # 环形高地
                mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), self.affine_matrices['height_g'])
                x_c = max(int(mapped_point[0][0][0]), 0)
                y_c = max(int(mapped_point[0][0][1]), 0)
                x_c = min(x_c, self.map_width)
                y_c = min(y_c, self.map_height)
                
                color = self.mask_image[y_c, x_c]
                if color[0] > color[2] and color[0] > color[1]:
                    height_level = "height_g"
                    X_M, Y_M = x_c, y_c
                else:
                    # 默认使用R型高地
                    height_level = "height_r"
                    X_M, Y_M = x_c, y_c
        
        return X_M, Y_M, height_level
    
    def pnp_precise_localization(self, bbox):
        """
        第二阶段: PnP精确3D定位
        基于装甲板几何关系和相机模型进行精确位姿求解
        """
        left, top, w, h = bbox
        
        # 从边界框计算四个角点（假设装甲板是矩形）
        bbox_2d_points = [
            [left, top],              # 左上
            [left + w, top],          # 右上
            [left + w, top + h],      # 右下  
            [left, top + h]           # 左下
        ]
        
        # 使用PnP求解3D位置
        success, camera_position = self.pnp_localizer.solve_pnp(bbox_2d_points)
        
        if success:
            # 将相机坐标系中的3D位置投影到地图
            map_x, map_y = self.pnp_localizer.project_to_map(camera_position, self.map_width, self.map_height)
            return True, map_x, map_y, camera_position
        else:
            return False, 0, 0, (0, 0, 0)
    
    def kalman_filter_smoothing(self, name, measured_x, measured_y):
        """
        第三阶段: 卡尔曼滤波平滑和预测
        处理急刹车、急转弯等非线性运动
        """
        if name not in self.filters:
            # 为新目标创建EKF
            self.filters[name] = ExtendedKalmanFilter()
            # 初始化状态
            self.filters[name].state[0, 0] = measured_x
            self.filters[name].state[1, 0] = measured_y
        else:
            # 更新EKF
            measurement = np.array([measured_x, measured_y])
            self.filters[name].predict()
            self.filters[name].update(measurement)
        
        # 获取滤波后的位置
        filtered_x, filtered_y = self.filters[name].get_position()
        return filtered_x, filtered_y
    
    def localize(self, name, bbox, camera_point):
        """
        完整的多级定位流程
        """
        # 1. 仿射变换粗略定位
        affine_x, affine_y, height_level = self.affine_transform_localization(camera_point)
        
        # 2. PnP精确3D定位
        pnp_success, pnp_x, pnp_y, camera_position = self.pnp_precise_localization(bbox)
        
        if pnp_success:
            # 使用PnP结果作为测量值
            measured_x, measured_y = pnp_x, pnp_y
            localization_method = "PnP"
        else:
            # PnP失败时使用仿射变换结果
            measured_x, measured_y = affine_x, affine_y
            localization_method = "Affine"
        
        # 3. 卡尔曼滤波平滑
        filtered_x, filtered_y = self.kalman_filter_smoothing(name, measured_x, measured_y)
        
        # 记录定位信息（用于调试）
        localization_info = {
            'method': localization_method,
            'affine': (affine_x, affine_y),
            'pnp': (pnp_x, pnp_y) if pnp_success else None,
            'filtered': (filtered_x, filtered_y),
            'height_level': height_level,
            'camera_position': camera_position if pnp_success else None
        }
        
        return filtered_x, filtered_y, localization_info

# =============================================================================
# 滑动窗口滤波器（作为备用）
# =============================================================================

class SlidingWindowFilter:
    """滑动窗口滤波器  作为卡尔曼滤波的备用方案"""
    def __init__(self, window_size=5, max_inactive_time=2.0, threshold=100000.0):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.threshold = threshold
        self.windows = {}
        self.last_update = {}

    def add_data(self, name, x, y):
        if name not in self.windows:
            self.windows[name] = deque(maxlen=self.window_size)

        # 异常值检测
        if len(self.windows[name]) > 0:
            last_x, last_y = self.windows[name][-1]
            if (x - last_x) ** 2 + (y - last_y) ** 2 > self.threshold:
                return

        self.windows[name].append((x, y))
        self.last_update[name] = time.time()

    def get_all_data(self):
        current_time = time.time()
        filtered = {}

        # 清理过期数据
        to_remove = []
        for name in self.windows:
            if current_time - self.last_update.get(name, 0) > self.max_inactive_time:
                to_remove.append(name)
                guess_list[name] = True

        for name in to_remove:
            del self.windows[name]
            del self.last_update[name]

        # 计算窗口均值
        for name, window in self.windows.items():
            if len(window) >= self.window_size:
                x_avg = sum(p[0] for p in window) / len(window)
                y_avg = sum(p[1] for p in window) / len(window)
                filtered[name] = (x_avg, y_avg)
                guess_list[name] = False

        return filtered

# 创建多级定位系统
localization_system = MultiLevelLocalization(
    camera_matrix, dist_coeffs, armor_3d_points,
    M_ground, M_height_r, M_height_g, mask_image
)

# 创建备用滑动窗口滤波器
backup_filter = SlidingWindowFilter(window_size=5, max_inactive_time=2.0)

# =============================================================================
# 相机系统
# =============================================================================

def hik_camera_get():
    """海康相机图像获取线程"""
    global camera_image
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # 枚举设备
    while 1:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
        if deviceList.nDeviceNum == 0:
            print("find no device!")
        else:
            print("Find %d devices!" % deviceList.nDeviceNum)
            break

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    
    nConnectionNum = '0'
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # 创建相机实例
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    print(get_Value(cam, param_type="float_value", node_name="ExposureTime"),
          get_Value(cam, param_type="float_value", node_name="Gain"),
          get_Value(cam, param_type="enum_value", node_name="TriggerMode"),
          get_Value(cam, param_type="float_value", node_name="AcquisitionFrameRate"))

    # 设置设备参数
    set_Value(cam, param_type="float_value", node_name="ExposureTime", node_value=user_ExposureTime)
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=user_Gain)
    
    # 开启设备取流
    start_grab_and_get_data_size(cam)
    stParam = MVCC_INTVALUE_EX()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
        
    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(pData)
            # 处理海康相机的图像格式为OPENCV处理的格式
            camera_image = image_control(data=image, stFrameInfo=stFrameInfo)
        else:
            print("no data[0x%x]" % ret)

def video_capture_get():
    """USB相机图像获取线程"""
    global camera_image
    cam = cv2.VideoCapture(1)
    while True:
        ret, img = cam.read()
        if ret:
            camera_image = img
            time.sleep(0.016)  # 60fps

# =============================================================================
# 串口通信系统
# =============================================================================

def ser_send():
    """串口发送线程"""
    if not ser1:
        print("串口未启用，发送线程退出")
        return
        
    seq = 0
    global chances_flag
    global guess_value
    global guess_table
    global guess_predict
    
    # 单点预测时间
    guess_time = {'B1': 0, 'B2': 0, 'B7': 0, 'R1': 0, 'R2': 0, 'R7': 0}
    
    # 预测点索引
    guess_index = {'B1': 0, 'B2': 0, 'B7': 0, 'R1': 0, 'R2': 0, 'R7': 0}

    # 发送蓝方机器人坐标
    def send_point_B(send_name, all_filter_data):
        # 转换为地图坐标系
        filtered_xyz = (2800 - all_filter_data[send_name][1], all_filter_data[send_name][0])
        # 转换为裁判系统单位M
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
        return ser_x, ser_y

    # 发送红方机器人坐标
    def send_point_R(send_name, all_filter_data):
        # 转换为地图坐标系
        filtered_xyz = (all_filter_data[send_name][1], 1500 - all_filter_data[send_name][0])
        # 转换为裁判系统单位M
        ser_x = int(filtered_xyz[0]) * 10 / 10
        ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
        return ser_x, ser_y

    # 发送盲区预测点坐标
    def send_point_guess(send_name, guess_time_limit):
        # 进度未满 and 预测进度没有涨 and 超过单点预测时间上限，同时满足则切换另一个点预测
        if (guess_value_now.get(send_name) < 120 and 
            guess_value_now.get(send_name) - guess_value.get(send_name) <= 0 and 
            time.time() - guess_time.get(send_name) >= guess_time_limit):
            
            guess_predict[send_name].get_points(send_name)
            points = guess_table.get(send_name)
            if points:
                guess_index[send_name] = (guess_index[send_name] + 1) % len(points)
            guess_time[send_name] = time.time()
            
        if guess_value_now.get(send_name) - guess_value.get(send_name) > 0:
            guess_time[send_name] = time.time()
            
        return (guess_table.get(send_name)[guess_index.get(send_name)][0],
                guess_table.get(send_name)[guess_index.get(send_name)][1])

    time_s = time.time()
    target_last = 0  # 上一帧的飞镖目标
    update_time = 0  # 上次预测点更新时间
    send_count = 0  # 信道占用数，上限为4
    
    send_map = {
        "R1": (0, 0), "R2": (0, 0), "R3": (0, 0), "R4": (0, 0), 
        "R5": (0, 0), "R6": (0, 0), "R7": (0, 0),
        "B1": (0, 0), "B2": (0, 0), "B3": (0, 0), "B4": (0, 0),
        "B5": (0, 0), "B6": (0, 0), "B7": (0, 0)
    }
    
    while True:
        send_count = 0  # 重置信道占用数
        try:
            # 使用滑动窗口滤波器获取数据（作为多级定位系统的接口）
            all_filter_data = backup_filter.get_all_data()
            
            if state == 'R':
                # 处理蓝方机器人
                for robot in ['B1', 'B2', 'B3', 'B4', 'B5']:
                    if not guess_list.get(robot):
                        if all_filter_data.get(robot, False):
                            send_map[robot] = send_point_B(robot, all_filter_data)
                    else:
                        send_map[robot] = (0, 0)

                # 哨兵特殊处理
                if guess_list.get('B7'):
                    send_map['B7'] = send_point_guess('B7', 4.7)  # guess_time_limit = 4.7
                else:
                    if all_filter_data.get('B7', False):
                        send_map['B7'] = send_point_B('B7', all_filter_data)

            elif state == 'B':
                # 处理红方机器人
                for robot in ['R1', 'R2', 'R3', 'R4', 'R5']:
                    if not guess_list.get(robot):
                        if all_filter_data.get(robot, False):
                            send_map[robot] = send_point_R(robot, all_filter_data)
                    else:
                        send_map[robot] = (0, 0)

                # 哨兵特殊处理
                if guess_list.get('R7'):
                    send_map['R7'] = send_point_guess('R7', 4.7)
                else:
                    if all_filter_data.get('R7', False):
                        send_map['R7'] = send_point_R('R7', all_filter_data)

            # 发送数据包
            ser_data = build_data_radar_all(send_map, state)
            packet, seq = build_send_packet(ser_data, seq, [0x03, 0x05])
            ser1.write(packet)
            time.sleep(0.2)
            print(send_map, seq)
            
            # 超过单点预测时间上限，更新上次预测的进度
            if time.time() - update_time > 4.7:
                update_time = time.time()
                if state == 'R':
                    guess_value['B1'] = guess_value_now.get('B1')
                    guess_value['B2'] = guess_value_now.get('B2')
                    guess_value['B7'] = guess_value_now.get('B7')
                else:
                    guess_value['R1'] = guess_value_now.get('R1')
                    guess_value['R2'] = guess_value_now.get('R2')
                    guess_value['R7'] = guess_value_now.get('R7')

            # 判断飞镖的目标是否切换，切换则尝试发动双倍易伤
            if target != target_last and target != 0:
                target_last = target
                # 有双倍易伤机会，并且当前没有在双倍易伤
                if double_vulnerability_chance > 0 and opponent_double_vulnerability == 0:
                    time_e = time.time()
                    # 发送时间间隔为10秒
                    if time_e - time_s > 10:
                        print("请求双倍触发")
                        data = build_data_decision(chances_flag, state)
                        packet, seq = build_send_packet(data, seq, [0x03, 0x01])
                        ser1.write(packet)
                        print("请求成功", chances_flag)
                        # 更新标志位
                        chances_flag += 1
                        if chances_flag >= 3:
                            chances_flag = 1
                        time_s = time.time()
                        
        except Exception as r:
            print('串口发送错误 %s' % (r))

def ser_receive():
    """裁判系统串口接收线程"""
    if not ser1:
        print("串口未启用，接收线程退出")
        return
        
    global progress_list
    global double_vulnerability_chance
    global opponent_double_vulnerability
    global target
    
    progress_cmd_id = [0x02, 0x0C]  # 雷达标记进度的命令码
    vulnerability_cmd_id = [0x02, 0x0E]  # 双倍易伤次数和触发状态
    target_cmd_id = [0x01, 0x05]  # 飞镖目标
    
    buffer = b''  # 初始化缓冲区
    
    while True:
        try:
            # 从串口读取数据
            received_data = ser1.read_all()
            buffer += received_data

            # 查找帧头（SOF）的位置
            sof_index = buffer.find(b'\xA5')

            while sof_index != -1:
                # 如果找到帧头，尝试解析数据包
                if len(buffer) >= sof_index + 5:
                    # 从帧头开始解析数据包
                    packet_data = buffer[sof_index:]

                    # 查找下一个帧头的位置
                    next_sof_index = packet_data.find(b'\xA5', 1)

                    if next_sof_index != -1:
                        # 如果找到下一个帧头，说明当前帧头到下一个帧头之间是一个完整的数据包
                        packet_data = packet_data[:next_sof_index]
                    else:
                        break

                    # 解析数据包
                    progress_result = receive_packet(packet_data, progress_cmd_id, info=False)
                    vulnerability_result = receive_packet(packet_data, vulnerability_cmd_id, info=False)
                    target_result = receive_packet(packet_data, target_cmd_id, info=False)
                    
                    # 更新裁判系统数据
                    if progress_result is not None:
                        received_cmd_id1, received_data1, received_seq1 = progress_result
                        progress_list = list(received_data1)
                        if state == 'R':
                            guess_value_now['B1'] = progress_list[0]
                            guess_value_now['B2'] = progress_list[1]
                            guess_value_now['B7'] = progress_list[5]
                        else:
                            guess_value_now['R1'] = progress_list[0]
                            guess_value_now['R2'] = progress_list[1]
                            guess_value_now['R7'] = progress_list[5]
                            
                    if vulnerability_result is not None:
                        received_cmd_id2, received_data2, received_seq2 = vulnerability_result
                        received_data2 = list(received_data2)[0]
                        double_vulnerability_chance, opponent_double_vulnerability = Radar_decision(received_data2)
                        
                    if target_result is not None:
                        received_cmd_id3, received_data3, received_seq3 = target_result
                        target = (list(received_data3)[1] & 0b1100000) >> 5

                    # 从缓冲区中移除已解析的数据包
                    buffer = buffer[sof_index + len(packet_data):]
                    # 继续寻找下一个帧头的位置
                    sof_index = buffer.find(b'\xA5')
                else:
                    break
                    
        except Exception as e:
            print(f"串口接收错误: {e}")
            
        time.sleep(0.5)

# =============================================================================
# 识别系统
# =============================================================================

# 加载模型，实例化机器人检测器和装甲板检测器
weights_path = 'car.onnx'  # 机器人检测模型
weights_path_next = 'armor.onnx'  # 装甲板检测模型

detector = YOLOv11Detector(weights_path, data='car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14, ui=True)
detector_next = YOLOv11Detector(weights_path_next, data='armor.yaml', conf_thres=0.50, iou_thres=0.2, max_det=1, ui=True)

# =============================================================================
# 初始化系统
# =============================================================================

# 串口初始化
if USART:
    try:
        ser1 = serial.Serial(user_com, 115200, timeout=1)
        print(f"串口已连接：{user_com}")
    except Exception as e:
        print(f"串口连接失败：{str(e)}")
        ser1 = None
        USART = False
else:
    ser1 = None
    print("串口功能已禁用")

# 启动串口线程
if USART:
    thread_receive = threading.Thread(target=ser_receive, daemon=True)
    thread_receive.start()
    thread_list = threading.Thread(target=ser_send, daemon=True)
    thread_list.start()

camera_image = None

# 相机初始化
if user_mode == 'test':
    camera_image = cv2.imread(user_img_test)
elif user_mode == 'hik':
    # 海康相机图像获取线程
    from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, get_Value, image_control
    if sys.platform.startswith("win"):
        from MvImport.MvCameraControl_class import *
    else:
        from MvImport_Linux.MvCameraControl_class import *
    thread_camera = threading.Thread(target=hik_camera_get, daemon=True)
    thread_camera.start()
elif user_mode == 'video':
    # USB相机图像获取线程
    thread_camera = threading.Thread(target=video_capture_get, daemon=True)
    thread_camera.start()

# 等待图像就绪
while camera_image is None:
    print("等待图像。。。")
    time.sleep(0.5)

# 获取相机图像的画幅，限制点不超限
img0 = camera_image.copy()
img_y = img0.shape[0]
img_x = img0.shape[1]
print(f"图像尺寸: {img0.shape}")

# =============================================================================
# 主循环
# =============================================================================

while True:
    # 刷新裁判系统信息UI图像
    information_ui_show = information_ui.copy()
    map = map_backup.copy()
    det_time = 0
    
    # 获取当前帧图像
    img0 = camera_image.copy()
    ts = time.time()

    # 第一层神经网络识别  检测机器人
    result0 = detector.predict(img0)
    det_time += 1
    
    for detection in result0:
        cls, xywh, conf = detection
        if cls == 'car':  # 只处理机器人类别
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            
            # ROI出机器人区域进行第二阶段的装甲板检测
            cropped = camera_image[top:top + h, left:left + w]
            cropped_img = np.ascontiguousarray(cropped)
            
            # 第二层神经网络识别  检测装甲板
            result_n = detector_next.predict(cropped_img)
            det_time += 1
            
            if result_n:
                # 叠加第二次检测结果到原图的对应位置
                img0[top:top + h, left:left + w] = cropped_img

                for detection1 in result_n:
                    cls, xywh, conf = detection1
                    if cls:  # 所有装甲板都处理
                        x, y, w, h = xywh
                        x = x + left
                        y = y + top

                        # =============================================================================
                        # 多级定位系统开始
                        # =============================================================================
                        
                        # 原图中装甲板的中心下沿作为待仿射变化的点
                        camera_point = np.array([[[min(x + 0.5 * w, img_x), min(y + 1.5 * h, img_y)]]],
                                                dtype=np.float32)
                        
                        # 使用多级定位系统进行定位
                        bbox = [left, top, w, h]  # 边界框信息
                        
                        try:
                            # 执行多级定位
                            filtered_x, filtered_y, loc_info = localization_system.localize(
                                cls, bbox, camera_point
                            )
                            
                            # 输出定位信息（用于调试）
                            print(f"定位结果 - {cls}: 方法={loc_info['method']}, "
                                  f"高度层={loc_info['height_level']}, "
                                  f"最终位置=({filtered_x:.1f}, {filtered_y:.1f})")
                            
                            # 将定位结果添加到备用滤波器
                            backup_filter.add_data(cls, filtered_x, filtered_y)
                            
                        except Exception as e:
                            print(f"定位错误 {cls}: {e}")
                            # 定位失败时使用传统仿射变换
                            mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
                            x_c = max(int(mapped_point[0][0][0]), 0)
                            y_c = max(int(mapped_point[0][0][1]), 0)
                            x_c = min(x_c, width)
                            y_c = min(y_c, height)
                            backup_filter.add_data(cls, x_c, y_c)
                        
                        # =============================================================================
                        # 多级定位系统结束
                        # =============================================================================

    # 获取所有识别到的机器人坐标（从备用滤波器）
    all_filter_data = backup_filter.get_all_data()
    
    # 在地图上绘制敌方机器人
    if all_filter_data != {}:
        for name, xyxy in all_filter_data.items():
            if xyxy is not None:
                if name[0] == "R":
                    color_m = (0, 0, 255)  # 红色
                else:
                    color_m = (255, 0, 0)  # 蓝色
                    
                if state == 'R':
                    filtered_xyz = (2800 - xyxy[1], xyxy[0])  # 缩放坐标到地图图像
                else:
                    filtered_xyz = (xyxy[1], 1500 - xyxy[0])  # 缩放坐标到地图图像
                    
                # 只绘制敌方阵营的机器人
                if name[0] != state:
                    cv2.circle(map, (int(filtered_xyz[0]), int(filtered_xyz[1])), 15, color_m, -1)  # 绘制圆
                    cv2.putText(map, str(name),
                                (int(filtered_xyz[0]) - 5, int(filtered_xyz[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                    ser_x = int(filtered_xyz[0]) * 10 / 10
                    ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
                    cv2.putText(map, "(" + str(ser_x) + "," + str(ser_y) + ")",
                                (int(filtered_xyz[0]) - 100, int(filtered_xyz[1]) + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    # 计算帧率
    te = time.time()
    t_p = te - ts
    print("fps:", 1 / t_p)

    # UI显示部分
    _ = draw_information_ui(progress_list, state, information_ui_show)
    cv2.putText(information_ui_show, "vulnerability_chances: " + str(double_vulnerability_chance),
                (10, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(information_ui_show, "vulnerability_Triggering: " + str(opponent_double_vulnerability),
                (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('information_ui', information_ui_show)
    
    map_show = cv2.resize(map, (600, 320))
    cv2.imshow('map', map_show)
    
    img0 = cv2.resize(img0, (1300, 900))
    cv2.imshow('img', img0)

    key = cv2.waitKey(1)
    if key == 27:  # ESC键退出
        break

# 清理资源
if 'ser1' in locals() and ser1:
    ser1.close()
cv2.destroyAllWindows()