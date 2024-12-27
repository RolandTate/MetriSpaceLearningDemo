import random
import os
import json
from tools import generate_data, minkowski_distance_factory, pivotSelectionRand

# 示例数据生成，值限制为整数

# Pivot Table 数据结构
class PivotTable:
    def __init__(self, data, pivots, distance_function):
        """
        初始化 Pivot Table 数据结构
        :param data: 数据集
        :param pivots: 支撑点集合
        :param distance_function: 距离函数，用于计算数据点和支撑点的距离
        """
        self.data = data
        self.pivots = pivots
        self.distance = [
            [distance_function(pivot, point) for point in data] for pivot in pivots
        ]

# Pivot Table 范围查询算法
def PTRangeSearch(pivot_table, distance_function, query_point, radius):
    """
    Pivot Table 的范围查询算法
    :param pivot_table: PivotTable 实例
    :param distance_function: 距离函数，用于计算支撑点与查询点的距离
    :param query_point: 查询点
    :param radius: 查询半径
    :return: 查询结果集
    """
    result = []  # 初始化结果集
    pivot_distance = []  # 支撑点与查询点的距离
    distance_count = 0

    # Step 1: 计算每个支撑点与查询点的距离，并判断是否是查询结果
    for pivot in pivot_table.pivots:
        dist = distance_function(pivot, query_point)  # 计算距离
        distance_count += 1
        pivot_distance.append(dist)
        if dist <= radius:  # 检查是否满足范围条件
            result.append(pivot)

    # Step 2: 处理每个数据对象
    for j, point in enumerate(pivot_table.data):
        done = False  # 标记当前数据对象是否被处理

        for i in range(len(pivot_table.pivots)):
            # 包含规则
            if pivot_distance[i] + pivot_table.distance[i][j] <= radius:
                result.append(point)
                done = True
                break

            # 排除规则
            if abs(pivot_distance[i] - pivot_table.distance[i][j]) > radius:
                done = True
                break

        # 如果无法排除或直接判定，则进行直接距离计算
        if not done:
            if distance_function(point, query_point) <= radius:
                distance_count += 1
                result.append(point)

    print(f'number of data points {len(pivot_table.data)}, distance count {distance_count} times')

    return result

if __name__ == "__main__":
    # 参数
    num_data_points = 10  # 数据点数量
    num_pivots = 1         # 支撑点数量
    dimensions = 1         # 数据点维度
    query_point = [8]  # 查询点
    search_radius = 15     # 查询半径
    file_path = "./data/"   # 文件路径
    data_file_name = f"data_{num_data_points}points_{dimensions}dimensions.json"  # 数据点文件名

    # 距离函数
    minkowski_t = 1  # 闵可夫斯基距离的参数 t or float('inf')
    minkowski_distance = minkowski_distance_factory(t=minkowski_t)

    if len(query_point) != dimensions:
        raise ValueError(f"Query point dimension ({len(query_point)}) does not match data dimensions ({dimensions}).")

    # 生成或加载数据点
    data = generate_data(
        num_points=num_data_points,
        dimensions=dimensions,
        save_to_file=True,
        file_name=data_file_name,
        file_path=file_path
    )

    # 随机选取支撑点
    pivots = pivotSelectionRand(data, num_pivots)
    # 手动移除支撑点 VP
    data = [x for x in data if x not in pivots]

    # 构建 Pivot Table
    pivot_table = PivotTable(data, pivots, minkowski_distance)

    # 打印部分数据
    print("Data points (first 5):", data[:5])
    print("Pivot points:", pivots)
    print("Query point:", query_point)
    print("Search radius:", search_radius)

    # 使用范围查询算法
    result = PTRangeSearch(pivot_table, minkowski_distance, query_point, search_radius)

    # 输出查询结果
    print("Query result (points within radius):")
    print(result)


