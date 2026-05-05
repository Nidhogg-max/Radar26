# test_hik.py
import sys
import os
import time
import cv2
import numpy as np
import ctypes
from ctypes import *

# ==========================================================
# 1. 动态加载 SDK 路径 (与 main.py 保持一致)
# ==========================================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport_Linux"))

try:
    from MvCameraControl_class import *
except ImportError as e:
    print("❌ 错误: 无法导入 MvCameraControl_class。")
    print("   请确保 'MvImport_Linux' 文件夹在当前目录下。")
    print(f"   详细报错: {e}")
    sys.exit(1)


def main():
    print("🚀 开始海康相机独立测试程序...")

    # 2. 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print(f"❌ 枚举设备失败! ret[0x{ret:x}]")
        return

    if deviceList.nDeviceNum == 0:
        print("❌ 未发现任何设备! 请检查 USB 连接。")
        return
    print(f"✅ 发现 {deviceList.nDeviceNum} 个设备")

    # 3. 创建相机实例并连接第一个设备
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print(f"❌ 创建句柄失败! ret[0x{ret:x}]")
        return

    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print(f"❌ 打开设备失败! ret[0x{ret:x}] (可能是权限问题，请用 sudo -E 运行)")
        return
    print("✅ 相机连接成功")

    # 4. 配置基础参数 (关闭触发模式，确保连续出图)
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    ret = cam.MV_CC_SetFloatValue("ExposureTime", 5000.0)  # 曝光 5000us
    ret = cam.MV_CC_SetFloatValue("Gain", 10.0)  # 增益 10

    # 获取负载大小
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    nPayloadSize = stParam.nCurValue

    # 5. 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print(f"❌ 开始取流失败! ret[0x{ret:x}]")
        return
    print("🎬 相机已启动，按 'q' 键退出...")

    # 数据缓存
    data_buf = (c_ubyte * nPayloadSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    try:
        while True:
            # 6. 获取一帧直接数据
            # 这里的 pData 是 ctypes 指针
            ret = cam.MV_CC_GetOneFrameTimeout(byref(data_buf), nPayloadSize, stFrameInfo, 1000)

            if ret == 0:
                # 获取成功，开始转换图像格式
                pData = (c_ubyte * stFrameInfo.nFrameLen).from_address(addressof(data_buf))
                image_data = np.frombuffer(pData, dtype=np.uint8)

                # 处理像素格式 (BayerRG -> RGB, 或者 Mono -> RGB)
                # 这里做简单处理，为了保证通用性，可能需要 ConvertPixelType
                # 但如果相机输出已经是非压缩格式，通常可以直接 reshape

                # 尝试 Reshape，海康相机默认通常是 BayerRG8 或 Mono8
                # 为了简化显示，我们直接使用 OpenCV 解码或转换
                # 这里假设是 BayerRG8 (常用彩色工业相机) -> 需要转码
                # 为了稳定性，我们在这里使用 SDK 自带的转换功能转为 RGB

                stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
                memset(byref(stConvertParam), 0, sizeof(stConvertParam))
                stConvertParam.nWidth = stFrameInfo.nWidth
                stConvertParam.nHeight = stFrameInfo.nHeight
                stConvertParam.pSrcData = data_buf
                stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
                stConvertParam.enSrcPixelType = stFrameInfo.enPixelType

                # 目标：BGR (OpenCV 格式)
                stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed

                # 准备输出 buffer
                nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight * 3
                img_buff = (c_ubyte * nConvertSize)()
                stConvertParam.pDstBuffer = img_buff
                stConvertParam.nDstBufferSize = nConvertSize

                ret_conv = cam.MV_CC_ConvertPixelType(stConvertParam)

                if ret_conv == 0:
                    # 转换成功，转 numpy
                    cdll_img = (c_ubyte * nConvertSize).from_address(addressof(img_buff))
                    image = np.frombuffer(cdll_img, dtype=np.uint8).reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, 3)

                    # 显示
                    cv2.imshow("HIK Camera Test", image)
                else:
                    print(f"图像转换失败 ret[0x{ret_conv:x}]，如果是Mono相机不需要转换")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"⚠️ 获取帧超时或失败: ret[0x{ret:x}]")

    except KeyboardInterrupt:
        pass
    finally:
        # 7. 关闭与清理
        print("\n🛑 正在停止相机...")
        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        cv2.destroyAllWindows()
        print("✅ 退出完成")


if __name__ == "__main__":
    main()