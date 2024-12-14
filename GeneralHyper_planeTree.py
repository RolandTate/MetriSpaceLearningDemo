import random
import math
from tools import generate_data, minkowski_distance_factory, pivotSelection
from Pivot_Table import PivotTable, PTRangeSearch

# GHT 树内部节点类
class GHTInternalNode:
    def __init__(self, c1, c2, left, right):
        self.c1 = c1  # 支撑点 c1
        self.c2 = c2  # 支撑点 c2
        self.left = left  # 左子树
        self.right = right  # 右子树

# 批量构建 GHT 树
def GHTBulkload(data, MaxLeafSize, distance_function):
    """
    批量构建 GHT 树
    :param data: 数据点集合
    :param MaxLeafSize: 叶子节点大小上界
    :param distance_function: 距离函数
    :return: GH 树根节点
    """
    if 0 == len(data):
        return None
    # 当数据量小于等于 MaxLeafSize 时，构建 Pivot Table 作为叶子节点
    if len(data) <= MaxLeafSize:
        pivot = pivotSelection(data, 1)
        # 手动移除支撑点 VP
        data = [x for x in data if x not in pivot]
        return PivotTable(data, pivot, distance_function)  # 构建 PivotTable

    # 选择两个支撑点（这里简单随机选择）
    c1, c2 = pivotSelection(data, 2)

    # 手动移除支撑点 c1 和 c2
    data = [x for x in data if x != c1 and x != c2]

    # 根据与支撑点的距离划分数据点
    leftData, rightData = [], []
    for s in data:
        if distance_function(s, c1) <= distance_function(s, c2):
            leftData.append(s)
        else:
            rightData.append(s)

    # 处理空子树
    left = GHTBulkload(leftData, MaxLeafSize, distance_function)
    right = GHTBulkload(rightData, MaxLeafSize, distance_function)

    return GHTInternalNode(c1, c2, left, right)

# GH 树范围查询算法
def GHTRangeSearch(node, query_point, radius, distance_function):
    """
    GH 树的范围查询算法
    :param node: 当前查询的节点（GHTInternalNode 或 PivotTable）
    :param query_point: 查询点
    :param radius: 查询范围半径
    :param distance_function: 距离函数
    :return: 查询结果列表
    """
    # 如果当前节点是叶子节点
    if isinstance(node, PivotTable):
        return PTRangeSearch(node, distance_function, query_point, radius)

    # 初始化结果列表
    result = []

    # 检查支撑点 c1 和 c2
    if distance_function(query_point, node.c1) <= radius:
        result.append(node.c1)  # 支撑点是否是查询结果
    if distance_function(query_point, node.c2) <= radius:
        result.append(node.c2)  # 支撑点是否是查询结果

    # 判断是否需要搜索左子树
    if distance_function(query_point, node.c1) - distance_function(query_point, node.c2) <= 2 * radius:
        if node.left:
            result.extend(GHTRangeSearch(node.left, query_point, radius, distance_function))  # 合并左子树结果

    # 判断是否需要搜索右子树
    if distance_function(query_point, node.c2) - distance_function(query_point, node.c1) <= 2 * radius:
        if node.right:
            result.extend(GHTRangeSearch(node.right, query_point, radius, distance_function))  # 合并右子树结果

    return result

if __name__ == "__main__":
    # 参数
    num_data_points = 20  # 数据点数量
    dimensions = 2         # 数据点维度
    query_point = [50, 50]  # 查询点
    search_radius = 20      # 查询半径
    max_leaf_size = 5       # 叶子节点大小上界
    file_path = "./data/"   # 文件路径
    data_file_name = f"data_{num_data_points}points_{dimensions}dimensions.json"  # 数据点文件名

    # 距离函数
    minkowski_t = 2  # 闵可夫斯基距离的参数 t
    minkowski_distance = minkowski_distance_factory(t=minkowski_t)

    # 校验查询点维度
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

    # 构建 GHT 树
    ght_root = GHTBulkload(data, max_leaf_size, minkowski_distance)

    # 打印部分数据
    print("Data points (first 5):", data[:5])
    print("Query point:", query_point)
    print("Search radius:", search_radius)
    print(f'GHT.c1: {ght_root.c1}, GHT.c2: {ght_root.c2}')

    # 使用范围查询算法
    result = GHTRangeSearch(ght_root, query_point, search_radius, minkowski_distance)

    # 输出查询结果
    print("Query result (points within radius):")
    print(result)
