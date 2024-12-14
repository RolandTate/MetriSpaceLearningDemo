import matplotlib.pyplot as plt

# 设置点的位置
p1 = (2, 3)  # 支撑点 p1
p2 = (8, 5)  # 支撑点 p2
q = (5, 7)   # 查询点 q
r = 2        # 查询半径

# 定义切比雪夫距离范围的正方形
def chebyshev_bounds(point, query, radius):
    """
    计算切比雪夫距离范围的上界和下界区域
    :param point: 支撑点坐标 (x, y)
    :param query: 查询点坐标 (x, y)
    :param radius: 查询半径
    :return: 四个边界值 (x_min, x_max, y_min, y_max)
    """
    x_min = abs(point[0] - query[0]) - radius
    x_max = abs(point[0] - query[0]) + radius
    y_min = abs(point[1] - query[1]) - radius
    y_max = abs(point[1] - query[1]) + radius
    return x_min, x_max, y_min, y_max

# 计算 p1 和 p2 的切比雪夫范围
p1_bounds = chebyshev_bounds(p1, q, r)
p2_bounds = chebyshev_bounds(p2, q, r)

# 绘制正方形区域
plt.figure(figsize=(8, 8))

# p1 的 -r 和 +r 范围
plt.gca().add_patch(plt.Rectangle((p1_bounds[0], p1_bounds[2]),
                                  p1_bounds[1] - p1_bounds[0], p1_bounds[3] - p1_bounds[2],
                                  fill=None, edgecolor='blue', linewidth=1.5, label="p1 Range"))

# p2 的 -r 和 +r 范围
plt.gca().add_patch(plt.Rectangle((p2_bounds[0], p2_bounds[2]),
                                  p2_bounds[1] - p2_bounds[0], p2_bounds[3] - p2_bounds[2],
                                  fill=None, edgecolor='red', linewidth=1.5, label="p2 Range"))

# 标记点 p1, p2 和 q
plt.scatter(*p1, color='blue', label="p1", s=100)
plt.scatter(*p2, color='red', label="p2", s=100)
plt.scatter(*q, color='black', label="q", s=100)

# 标注点的位置
plt.text(p1[0], p1[1], "p1", fontsize=12, color="blue", ha='right')
plt.text(p2[0], p2[1], "p2", fontsize=12, color="red", ha='right')
plt.text(q[0], q[1], "q", fontsize=12, color="black", ha='right')

# 设置图形范围
plt.xlim(0, 12)
plt.ylim(0, 12)

# 添加网格和图例
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("切比雪夫距离的四个正方形区域")

# 显示图形
plt.show()
