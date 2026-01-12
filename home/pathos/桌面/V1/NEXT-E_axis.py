"""
@FileName:    NEXT-E坐标系.py
@Description: 描述
@Version:     1.0
@Software:    VS code
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

background_image_path = 'images/2025map.png'  # 背景图路径
background_extent = [0, 2800, 0, 1500]

point_radius = 25  # 点的大小
circle_radius_1 = 80  # 内圆半径（绿色）
circle_radius_2 = 160  # 外圆半径（黄色）
circle_linestyle = '--'
circle_linewidth = 1
annotation_offset = (10, -10)  # 文字偏移量
annotation_fontsize = 10  # 文字大小
annotation_bbox_style = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1)
guess_table = {
    "R1": [(1000, 400), (960, 1000), (1123, 1195), (800, 1225), (946, 1341), (457, 1232)],
    # "R2": [(200, 100), (900, 900), (900, 600), (1335, 821), (1469, 687)],
    # "R3": [(998, 1059), (1186, 1266), (2030, 184)],
    # "R4": [(998, 1059), (1186, 1266), (2030, 184)],
    # "R7": [(386, 812), (1356, 1093)],

    "B1": [(1821, 1092), (1851, 513), (1754, 403), (2050, 305), (1800, 200),(2294,225)],
    # "B2": [(2600, 1400), (1900, 636), (1900, 878), (1500, 750), (1410, 654)],
    # "B3": [(1814, 475), (784, 1372), (1646, 270)],
    # "B4": [(1460, 820), (1930, 400)],
    # "B7": [(2240, 870), (2240, 603)],
}
annotations = {
    "R1": ["吊射点1", "吊射点2", "隧道点", "高地吊射点1", "高地吊射点2", "高地吊射点3"],
    "R2": ["兑矿区", "银矿1", "银矿2", "金矿1", "金矿2"],
    "R3": ["英雄保护点1", "英雄保护点2", "sd"],
    "R4": [],
    "R7": [],

    "B1": ["吊射点1", "吊射点2", "隧道点", "高地吊射点1", "高地吊射点2", "高地吊射点3"],
    "B2": ["兑矿区", "银矿1", "银矿2", "金矿1", "金矿2"],
    "B3": ["英雄保护点1", "英雄保护点2", "sd"],
    "B4": [],
    "B7": [],
}

show_axes = True
x_label = 'X轴'
y_label = 'Y轴'
axis_color = 'black'
axis_linewidth = 2
axis_style = '-'


show_grid = True
# grid_color = 'gray'
grid_color = 'red'
grid_style = '-.'
grid_linewidth = 0.2
grid_spacing_x = 100  # x轴间距
grid_spacing_y = 100  # y轴间距

label_fontsize = 14
tick_fontsize = 10

fig, ax = plt.subplots(figsize=(14, 8))
img = mpimg.imread(background_image_path)
ax.imshow(img[::-1], extent=background_extent, origin='lower')
ax.set_xlim(background_extent[0], background_extent[1])
ax.set_ylim(background_extent[2], background_extent[3])
for key, points in guess_table.items():
    color = 'red' if key.startswith('R') else 'blue'
    label_list = annotations.get(key, [])

    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, s=point_radius, color=color, zorder=5)
        for radius, color_r in zip([circle_radius_1, circle_radius_2], ['green', 'yellow']):
            circle = plt.Circle(
                (x, y), radius,
                fill=False, linestyle=circle_linestyle,
                edgecolor=color_r, linewidth=circle_linewidth
            )
            ax.add_artist(circle)

        if i < len(label_list):
            label = f"{key}: {label_list[i]}"
            ax.annotate(
                label,
                xy=(x, y),
                xytext=annotation_offset,
                textcoords='offset points',
                fontsize=annotation_fontsize,
                color="#000000",
                bbox=annotation_bbox_style,
                zorder=6
            )

if show_axes:
    ax.set_xlabel(x_label, fontsize=label_fontsize, color=axis_color)
    ax.set_ylabel(y_label, fontsize=label_fontsize, color=axis_color)
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.spines['bottom'].set_linewidth(axis_linewidth)
    ax.spines['left'].set_linewidth(axis_linewidth)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

if show_grid:
    ax.set_xticks(range(int(background_extent[0]), int(background_extent[1]) + 1, grid_spacing_x))
    ax.set_yticks(range(int(background_extent[2]), int(background_extent[3]) + 1, grid_spacing_y))
    ax.grid(True, which='both', linestyle=grid_style, color=grid_color, linewidth=grid_linewidth)

plt.title('RMUC')
plt.tight_layout()
plt.show()

# 保存图像
# output_path = 'output_map_with_grid.png'
# dpi = 100
# fig.set_size_inches(28, 15)
# plt.savefig(output_path, dpi=dpi)
# print(f'保存成功：{output_path}')
