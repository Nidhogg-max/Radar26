import matplotlib.pyplot as plt
import numpy as np
import cv2

# --- 添加以下代码段来解决中文显示问题 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_ansi'] = False    # 用来正常显示负号
except Exception as e:
    print(f"字体设置失败: {e}")
# ----------------------------------------

# ... 以下是您原本的代码 ...
# image_points = np.array([...
image_points = np.array([
    [171, 303],
    [626, 304],
    [1064, 303],
    [104, 415],
    [626, 420],
    [1123, 424],
    [0, 547],
    [621, 550],
    [1180, 554]
])

world_points = np.array([
    [375, 375],   [750, 375],   [1125, 375],
    [375, 750],   [750, 750],   [1125, 750],
    [375, 1125],  [750, 1125],  [1125, 1125]
])

# 可视化点对
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 图像点
axes[0].scatter(image_points[:,0], image_points[:,1], c='red', s=100)
for i, (x, y) in enumerate(image_points):
    axes[0].text(x, y, f'Img{i+1}', fontsize=12, ha='center', va='bottom', color='blue')
axes[0].set_title('图像坐标点 (像素)')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].grid(True)
axes[0].invert_yaxis()  # 图像坐标系Y向下为正

# 地图点
axes[1].scatter(world_points[:,0], world_points[:,1], c='green', s=100)
for i, (x, y) in enumerate(world_points):
    axes[1].text(x, y, f'Map{i+1}', fontsize=12, ha='center', va='bottom', color='blue')
axes[1].set_title('地图坐标点 (1500×1500)')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].grid(True)
axes[1].set_xlim(0, 1500)
axes[1].set_ylim(0, 1500)

# 显示对应线
for i in range(9):
    axes[0].text(image_points[i,0], image_points[i,1], f'→ Map{i+1}',
                fontsize=9, ha='left', va='top', color='purple')
    axes[1].text(world_points[i,0], world_points[i,1], f'← Img{i+1}',
                fontsize=9, ha='right', va='top', color='purple')

plt.tight_layout()
plt.savefig('point_correspondence_check.png', dpi=150)
plt.show()

print("检查点对布局是否匹配：")
print("图像点应呈现类似网格布局")
print("地图点应是完美的3×3网格")