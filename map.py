import cv2
import numpy as np


# 方法1: 创建纯黑色图片
def create_black_image(width=1000, height=1000):
    """创建指定尺寸的纯黑色图片"""
    # 创建全0的数组，对应黑色
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    return black_image


# 方法2: 使用指定颜色创建
def create_color_image(width=1000, height=1000, color=(0, 0, 0)):
    """创建指定颜色和尺寸的图片"""
    # 创建一个全为指定颜色的图片
    image = np.full((height, width, 3), color, dtype=np.uint8)
    return image


# 方法3: 创建并显示图片
def create_and_display_black_image():
    """创建、显示并保存黑色图片"""
    width, height = 1000, 1000

    print("=" * 50)
    print("创建1500×1500纯黑色图片")
    print("=" * 50)

    # 创建黑色图片
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    # 打印图片信息
    print(f"图片尺寸: {width}×{height}")
    print(f"数组形状: {black_image.shape}")
    print(f"数据类型: {black_image.dtype}")
    print(f"最小值: {black_image.min()}, 最大值: {black_image.max()}")

    # 保存图片
    output_filename = "map.png"
    cv2.imwrite(output_filename, black_image)
    print(f"✅ 图片已保存: {output_filename}")

    # 显示图片（会缩小显示）
    # 先缩小到可显示的大小
    display_size = 600
    display_img = cv2.resize(black_image, (display_size, display_size))

    # 添加标题
    cv2.putText(display_img, "1500x1500 Black Image", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(display_img, f"Original: {width}x{height}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("1500x1500 Black Image", display_img)
    print("\n显示图片中... 按任意键关闭窗口")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return black_image, output_filename


# 方法4: 创建多种纯色图片
def create_various_images():
    """创建多种纯色图片"""
    width, height = 1000, 1000

    # 定义不同颜色
    colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "gray": (128, 128, 128)
    }

    print("\n" + "=" * 50)
    print("创建多种1500×1500纯色图片")
    print("=" * 50)

    created_files = []

    for color_name, color_bgr in colors.items():
        # 创建纯色图片
        image = np.full((height, width, 3), color_bgr, dtype=np.uint8)

        # 保存图片
        filename = f"{color_name}_1500x1500.png"
        cv2.imwrite(filename, image)

        created_files.append(filename)
        print(f"✅ 创建: {filename}")

    return created_files


# 主程序
if __name__ == "__main__":
    # 1. 创建并显示黑色图片
    print("🔧 OpenCV 纯色图片生成器")
    print("=" * 50)

    # 简单创建黑色图片
    black_img = create_black_image(1000, 1000)
    print("简单创建: 1500x1500黑色图片")

    # 完整创建流程
    img, filename = create_and_display_black_image()

    # 可选：创建多种颜色图片
    print("\n是否创建其他颜色图片？(y/n): ", end="")
    response = input().strip().lower()

    if response == 'y':
        files = create_various_images()
        print(f"\n总计创建 {len(files)} 个图片文件")

    print("\n✅ 程序完成")