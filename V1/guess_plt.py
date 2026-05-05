import math
import matplotlib.pyplot as plt
import numpy as np

d_factor = 0.01
cos_factor = 0.003
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


class Predict:
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
            d_score = math.exp(-distance * d_factor)

            # 分数值确定优先级
            score = cos_factor * cos_sim + (1 - cos_factor) * d_score
            scores.append((point, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [item[0] for item in scores]

    def get_points(self, name):
        # if self.flag:
        guess_table[name] = self.predict_point(guess_table.get(name))
        # self.flag = False


guess_predict = {
    "B1": Predict(),
    "B2": Predict(),
    "B3": Predict(),
    "B4": Predict(),
    "B7": Predict(),

    "R1": Predict(),
    "R2": Predict(),
    "R3": Predict(),
    "R4": Predict(),
    "R7": Predict()
}

l1 = [(400, 700), (600, 600)] # 消失点
l2 = guess_table['R2'] # 盲点
index = 1  # l1的索引

guess_predict['R2'].add_point(l1[0])
guess_predict['R2'].add_point(l1[1])

print('排序前：',guess_table['R2'])
guess_predict['R2'].get_points('R2')
print('排序后：',guess_table['R2'])


x1, y1 = zip(*l1)
x2, y2 = zip(*l2)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.figure(figsize=(10, 8))
plt.scatter(x1, y1, c='blue', s=80, label='消失点（两个）', zorder=5)
plt.scatter(x2, y2, c='red', s=80, label='盲点', zorder=5)
# 圆心
center = l1[index]
plt.scatter(center[0], center[1], c='green', s=120, marker='*', label='消失点', zorder=6)
# 绘制同心圆
max_radius = 0
for point in l2:
    radius = np.sqrt((center[0] - point[0]) ** 2 + (center[1] - point[1]) ** 2)
    max_radius = max(max_radius, radius)
    circle = plt.Circle(center, radius, fill=False, color='purple', linewidth=1.5)
    plt.gca().add_patch(circle)

plt.axis('equal')
margin = max_radius * 0.2
plt.xlim(center[0] - max_radius - margin, center[0] + max_radius + margin)
plt.ylim(center[1] - max_radius - margin, center[1] + max_radius + margin)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
