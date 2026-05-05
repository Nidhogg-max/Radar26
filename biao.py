import numpy as np
import cv2

# 您提供的图像坐标点
image_points = np.array([
    [250 , 161],
    [590 , 173],
    [920 , 174],
    [207 , 243],
    [582 , 254],
    [946 , 262],
    [125 , 341],
    [569 , 347],
    [965 , 353]
], dtype=np.float32)

# 300×300地图坐标点
world_points = np.array([
    [30, 30],  # 点1: 左上角
    [150, 30],  # 点2: 上中
    [270, 30],  # 点3: 右上角
    [30, 150],  # 点4: 左中
    [150, 150],  # 点5: 中心
    [270, 150],  # 点6: 右中
    [30, 270],  # 点7: 左下角
    [150, 270],  # 点8: 下中
    [270, 270]  # 点9: 右下角
], dtype=np.float32)

print("=" * 60)
print("🎯 300×300地图仿射变换矩阵标定系统")
print("=" * 60)

# 1. 标定点对验证
print("\n🔢 标定点对验证")
print("-" * 50)

# 图像坐标点分布
print("图像坐标点分布:")
print(f"  X范围: {image_points[:, 0].min():.0f} 到 {image_points[:, 0].max():.0f}")
print(f"  Y范围: {image_points[:, 1].min():.0f} 到 {image_points[:, 1].max():.0f}")
print(f"  中心点: ({image_points[:, 0].mean():.1f}, {image_points[:, 1].mean():.1f})")

# 地图坐标点分布
print("\n地图坐标点分布:")
print(f"  X范围: {world_points[:, 0].min():.0f} 到 {world_points[:, 0].max():.0f}")
print(f"  Y范围: {world_points[:, 1].min():.0f} 到 {world_points[:, 1].max():.0f}")
print(f"  中心点: ({world_points[:, 0].mean():.1f}, {world_points[:, 1].mean():.1f})")

# 2. 计算仿射变换矩阵
print("\n" + "=" * 60)
print("🔄 计算仿射变换矩阵")
print("=" * 60)

# 使用LMEDS方法计算2D仿射变换
M_affine, inliers = cv2.estimateAffine2D(image_points, world_points, method=cv2.LMEDS)

# 转换为3×3齐次坐标矩阵
M_homogeneous = np.vstack([M_affine, [0, 0, 1]])

print("计算得到的仿射变换矩阵 (3×3):")
print()
print("M_300x300 = np.array([")
for i in range(3):
    if i < 2:
        print(f"    [{M_homogeneous[i, 0]:.8f}, {M_homogeneous[i, 1]:.8f}, {M_homogeneous[i, 2]:.8f}],")
    else:
        print(f"    [{M_homogeneous[i, 0]:.8f}, {M_homogeneous[i, 1]:.8f}, {M_homogeneous[i, 2]:.8f}]")
print("])")

# 3. 标定精度验证
print("\n" + "=" * 60)
print("📊 标定精度验证")
print("=" * 60)

errors = []
max_error = 0
max_error_idx = 0
min_error = float('inf')
min_error_idx = 0

print("点对转换验证:")
for i in range(len(image_points)):
    # 将图像坐标转换为齐次坐标
    point = np.array([[[image_points[i, 0], image_points[i, 1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, M_homogeneous)

    x_pred, y_pred = mapped[0, 0, 0], mapped[0, 0, 1]
    x_real, y_real = world_points[i, 0], world_points[i, 1]

    # 计算误差
    error = np.sqrt((x_pred - x_real) ** 2 + (y_pred - y_real) ** 2)
    errors.append(error)

    # 更新最大误差
    if error > max_error:
        max_error = error
        max_error_idx = i

    # 更新最小误差
    if error < min_error:
        min_error = error
        min_error_idx = i

    print(f"点{i + 1}:")
    print(f"  图像坐标: ({image_points[i, 0]:.0f}, {image_points[i, 1]:.0f})")
    print(f"  预测坐标: ({x_pred:.2f}, {y_pred:.2f})")
    print(f"  实际坐标: ({x_real:.0f}, {y_real:.0f})")
    print(f"  转换误差: {error:.4f}像素")
    print()

# 精度统计
mean_error = np.mean(errors)
std_error = np.std(errors)
median_error = np.median(errors)

print("-" * 50)
print("📈 精度统计分析:")
print(f"  平均误差: {mean_error:.4f}像素")
print(f"  最大误差: {max_error:.4f}像素 (点{max_error_idx + 1})")
print(f"  最小误差: {min_error:.4f}像素 (点{min_error_idx + 1})")
print(f"  误差中位数: {median_error:.4f}像素")
print(f"  误差标准差: {std_error:.4f}像素")

# 精度评估
print("\n🔍 精度评估:")
if mean_error < 1.0:
    print("  ✅ 标定精度: 优秀 (平均误差 < 1像素)")
elif mean_error < 3.0:
    print("  ✅ 标定精度: 良好 (1像素 ≤ 平均误差 < 3像素)")
elif mean_error < 5.0:
    print("  ⚠️ 标定精度: 一般 (3像素 ≤ 平均误差 < 5像素)")
elif mean_error < 10.0:
    print("  ⚠️ 标定精度: 较差 (5像素 ≤ 平均误差 < 10像素)")
else:
    print("  ❌ 标定精度: 很差 (平均误差 ≥ 10像素)")

# 4. 矩阵参数分析
print("\n" + "=" * 60)
print("🔍 矩阵参数分析")
print("=" * 60)

a, b, c = M_homogeneous[0, 0], M_homogeneous[0, 1], M_homogeneous[0, 2]
d, e, f = M_homogeneous[1, 0], M_homogeneous[1, 1], M_homogeneous[1, 2]

# 计算变换参数
scale_x = np.sqrt(a * a + d * d)
scale_y = np.sqrt(b * b + e * e)
rotation = np.arctan2(d, a) * 180 / np.pi
shear = np.arctan2(b, a) * 180 / np.pi
scale_ratio = scale_x / scale_y
scale_diff_percent = (scale_ratio - 1) * 100

print("变换参数分解:")
print(f"  平移分量: tx = {c:.6f}, ty = {f:.6f}")
print(f"  X方向缩放: {scale_x:.6f}")
print(f"  Y方向缩放: {scale_y:.6f}")
print(f"  旋转角度: {rotation:.4f}°")
print(f"  剪切角度: {shear:.4f}°")

print("\n缩放比例分析:")
print(f"  缩放比例: X:{scale_x:.6f}, Y:{scale_y:.6f}")
print(f"  缩放比: {scale_ratio:.6f}")
print(f"  缩放差异: {scale_diff_percent:.2f}%")

# 缩放比例合理性检查
if abs(scale_diff_percent) > 20.0:
    print(f"  ⚠️ 警告: X和Y方向缩放比例差异较大 ({abs(scale_diff_percent):.1f}%)")
else:
    print(f"  ✅ X和Y方向缩放比例差异在合理范围内")

# 5. 保存标定矩阵
output_filename = "calibration_300x300_matrix.npy"
np.save(output_filename, M_homogeneous)
print(f"\n💾 标定矩阵已保存到: {output_filename}")

# 6. 关键点转换测试
print("\n" + "=" * 60)
print("🧪 关键点转换测试")
print("=" * 60)

test_points = [
    (image_points[4, 0], image_points[4, 1], "中心点(点5)", (150, 150)),
    (image_points[0, 0], image_points[0, 1], "左上点(点1)", (30, 30)),
    (image_points[2, 0], image_points[2, 1], "右上点(点3)", (270, 30)),
    (image_points[6, 0], image_points[6, 1], "左下点(点7)", (30, 270)),
    (image_points[8, 0], image_points[8, 1], "右下点(点9)", (270, 270))
]

print("关键点转换测试结果:")
for x, y, desc, (exp_x, exp_y) in test_points:
    point = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, M_homogeneous)

    x_map = mapped[0, 0, 0]
    y_map = mapped[0, 0, 1]
    error = np.sqrt((x_map - exp_x) ** 2 + (y_map - exp_y) ** 2)

    # 状态标记
    if error < 1.0:
        status = "✅"
    elif error < 3.0:
        status = "⚠️"
    else:
        status = "❌"

    print(f"{status} {desc}:")
    print(f"  图像坐标: ({x:.0f}, {y:.0f})")
    print(f"  地图坐标: ({x_map:.2f}, {y_map:.2f})")
    print(f"  期望坐标: ({exp_x}, {exp_y})")
    print(f"  转换误差: {error:.4f}像素")
    print()

# 7. 图像边界测试
print("\n" + "=" * 60)
print("🔬 图像边界测试")
print("=" * 60)

# 假设图像尺寸为1280×720
image_width, image_height = 1280, 720
boundary_points = [
    (0, 0, "图像左上角"),
    (image_width, 0, "图像右上角"),
    (0, image_height, "图像左下角"),
    (image_width, image_height, "图像右下角"),
    (image_width // 2, image_height // 2, "图像中心")
]

print("图像边界点映射测试:")
for x, y, desc in boundary_points:
    point = np.array([[[x, y]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(point, M_homogeneous)

    map_x, map_y = mapped[0, 0, 0], mapped[0, 0, 1]

    # 检查是否在300×300范围内
    in_range_x = 0 <= map_x <= 300
    in_range_y = 0 <= map_y <= 300
    in_range = in_range_x and in_range_y

    range_status = "✅" if in_range else "⚠️"

    print(f"{range_status} {desc}:")
    print(f"  图像坐标: ({x}, {y})")
    print(f"  地图坐标: ({map_x:.1f}, {map_y:.1f})")
    print(f"  是否在0-300范围内: {'是' if in_range else '否'}")
    if not in_range_x:
        print(f"  X坐标超出范围: {map_x:.1f}")
    if not in_range_y:
        print(f"  Y坐标超出范围: {map_y:.1f}")
    print()

# 8. 生成使用代码模板
print("=" * 60)
print("🚀 在您的代码中使用此标定矩阵")
print("=" * 60)

