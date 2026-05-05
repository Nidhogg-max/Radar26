import cv2
import numpy as np
import time
import sys
import os

# 导入检测模块
try:
    from detect_function_yolov11 import YOLOv11Detector, YOLOv11Classifier

    print("检测模块导入成功")
except ImportError as e:
    print(f"错误: 无法导入检测模块: {e}")
    sys.exit(1)

# ================= 配置区域 =================
# 视频文件路径
VIDEO_PATH = "/home/pathos/桌面/V1(1625.19)/V1/images/test2.mp4"

# 【新增配置】目标帧率
TARGET_FPS = 60
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 每一帧的理论时间间隔 (秒)


# ===========================================

def main():
    # 1. 初始化模型
    print("正在加载模型...")
    try:
        # 第一阶段：车体检测器
        stage1_weights = '/home/pathos/桌面/V1(1625.19)/V1/modelEEE/car_1280_best.engine'
        stage1_detector = YOLOv11Detector(
            weights_path=stage1_weights,
            img_size=1280,
            conf_thres=0.1,
            iou_thres=0.5,
            max_det=14,
            data='yaml/car.yaml',
            ui=True
        )

        # 第二阶段：装甲板检测器
        stage2_weights = '/home/pathos/桌面/V1(1625.19)/V1/modelEEE/arrmor_192_best.engine'
        stage2_detector = YOLOv11Detector(
            weights_path=stage2_weights,
            img_size=192,
            conf_thres=0.4,
            iou_thres=0.2,
            max_det=5,
            data='yaml/armor.yaml',
            ui=True
        )

        # 第三阶段：装甲板数字分类器
        stage3_weights = '/home/pathos/桌面/V1(1625.19)/V1/modelEEE/cls_64_best.engine'
        stage3_classifier = YOLOv11Classifier(
            weights_path=stage3_weights,
            img_size=64
        )
        print("模型加载完成")
    except Exception as e:
        print(f"模型加载失败，请检查路径: {e}")

    # 2. 打开视频流
    if os.path.exists(VIDEO_PATH):
        cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"正在读取视频文件: {VIDEO_PATH}")
    else:
        print(f"未找到视频文件 {VIDEO_PATH}，尝试打开默认摄像头...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("无法打开视频源")
        return

    # 3. 循环处理
    frame_cnt = 0
    t_start_loop = time.time()  # 初始化这一帧的开始时间

    while True:
        # 记录每帧循环开始的绝对时间
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("视频播放结束或无法读取帧")
            # 循环播放逻辑：如果需要循环，取消下面两行的注释
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # continue
            break

        frame_cnt += 1

        # 复制一份用于画图
        img_show = frame.copy()
        img_h, img_w = frame.shape[:2]

        # ==================== 第一阶段：车体检测 ====================
        cars = stage1_detector.predict(frame)

        for car_det in cars:
            cls, box, conf = car_det

            # 解析中心点坐标
            cx, cy, w, h = map(int, box)

            # 反算左上角坐标
            car_left = int(cx - w / 2)
            car_top = int(cy - h / 2)
            car_w, car_h = w, h

            # 画车体框 (青色)
            cv2.rectangle(img_show, (car_left, car_top),
                          (car_left + car_w, car_top + car_h), (255, 255, 0), 2)

            # ==================== ROI 安全裁剪 ====================
            y1 = max(0, car_top)
            x1 = max(0, car_left)
            y2 = min(img_h, car_top + car_h)
            x2 = min(img_w, car_left + car_w)

            if y2 > y1 and x2 > x1:
                car_roi = frame[y1:y2, x1:x2]
                car_roi_contiguous = np.ascontiguousarray(car_roi)

                # ==================== 第二阶段：装甲板检测 ====================
                armors = stage2_detector.predict(car_roi_contiguous)

                if armors:
                    for armor_det in armors:
                        a_cls, a_box, a_conf = armor_det

                        # 解析装甲板中心点并转回全局坐标
                        a_cx, a_cy, a_w, a_h = map(int, a_box)
                        a_left = int(a_cx - a_w / 2)
                        a_top = int(a_cy - a_h / 2)

                        global_a_left = x1 + a_left
                        global_a_top = y1 + a_top

                        cv2.rectangle(img_show, (global_a_left, global_a_top),
                                      (global_a_left + a_w, global_a_top + a_h), (255, 0, 255), 2)

                        # ==================== 第三阶段：数字分类 ====================
                        ay1 = max(0, a_top)
                        ax1 = max(0, a_left)
                        ay2 = min(car_roi_contiguous.shape[0], a_top + a_h)
                        ax2 = min(car_roi_contiguous.shape[1], a_left + a_w)

                        if ay2 > ay1 and ax2 > ax1:
                            armor_roi = car_roi_contiguous[ay1:ay2, ax1:ax2]
                            armor_roi_contiguous = np.ascontiguousarray(armor_roi)

                            if armor_roi_contiguous.size > 0:
                                classes = stage3_classifier.predict(armor_roi_contiguous, top_k=1)

                                if classes:
                                    res_cls, res_conf = classes[0]
                                    label_text = f"{res_cls} {res_conf:.2f}"
                                    cv2.putText(img_show, label_text,
                                                (global_a_left, global_a_top - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ================= 帧率稳定逻辑 (核心修改) =================
        # 1. 计算当前处理逻辑消耗的时间
        process_end_time = time.time()
        process_duration = process_end_time - loop_start_time

        # 2. 计算这一帧的实际 FPS (根据循环间隔)
        current_actual_fps = 1.0 / (process_end_time - t_start_loop) if (process_end_time - t_start_loop) > 0 else 0
        t_start_loop = process_end_time  # 更新上一帧结束时间

        # 3. 计算需要 sleep 多久才能达到目标 60 FPS
        # 如果处理太快 (process_duration < 0.016s)，我们就等多出来的这部分时间
        wait_time_seconds = FRAME_INTERVAL - process_duration

        cv2.putText(img_show, f"FPS: {current_actual_fps:.1f} / Target: {TARGET_FPS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Detection Test", img_show)

        # 4. 智能等待
        delay_ms = 1  # 默认最少等待1ms防止卡死UI
        if wait_time_seconds > 0:
            # 如果处理很快，剩余时间交给 waitKey
            delay_ms = int(wait_time_seconds * 1000)
            if delay_ms == 0: delay_ms = 1

        # 按 'q' 键退出
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()