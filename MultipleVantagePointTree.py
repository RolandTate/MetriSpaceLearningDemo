from tools import generate_data, minkowski_distance_factory, pivotSelectionRand, determineSplitRadius, getAllData
from Pivot_Table import PivotTable, PTRangeSearch
from PivotSelectionAlgorithm import maxVarianceSelection

# 按支撑点划分数据集
def split(data, vantage_point, num_regions, distance_function):
    distances = [(point, distance_function(point, vantage_point)) for point in data]
    distances.sort(key=lambda x: x[1])  # 按距离排序
    partition_size = len(data) // num_regions
    partitions = []
    for i in range(num_regions):
        start = i * partition_size
        end = start + partition_size if i < num_regions - 1 else len(data)
        partitions.append([x[0] for x in distances[start:end]])  # 划分数据
    return partitions

# MVP 树内部节点类
class MVPTInternalNode:
    def __init__(self, pivots, children, lowerBound, upperBound):
        self.pivots = pivots  # 支撑点
        self.children = children  # 子树
        self.lowerBound = lowerBound  # 每棵子树到每个支撑点的距离下界
        self.upperBound = upperBound  # 每棵子树到每个支撑点的距离上界

# MVP 树批建算法
def MVPTBulkload(data, MaxLeafSize, k, num_regions, distance_function):
    """
    MVP 树批建算法
    :param data: 数据点集合
    :param MaxLeafSize: 叶子节点大小上界
    :param k: 支撑点数量
    :param num_regions: 每个支撑点划分的区域数
    :param distance_function: 距离函数
    :return: VP 树根节点
    """
    if 0 == len(data):
        return None
    # 当数据量小于等于 MaxLeafSize 时，构建 Pivot Table 作为叶子节点
    if len(data) < k or len(data) <= MaxLeafSize:
        # 随机选择支撑点
        pivot = pivotSelectionRand(data, 1)
        # 手动移除支撑点 VP
        data = [x for x in data if x not in pivot]
        return PivotTable(data, pivot, distance_function)  # 构建 PivotTable

    # 选择支撑点
    VP = maxVarianceSelection(data, k, distance_function)

    # 移除支撑点
    data = [x for x in data if x not in VP]  # 移除支撑点
    partitions = [data]  # 初始化划分

    # 按支撑点划分数据集
    for i in range(k):
        newPartitions = []
        for par in partitions:
            if len(par) > 0:
                # 每个现子集基于当前支撑点划分成f个新子集
                newPartitions.extend(split(par, VP[i], num_regions, distance_function))
        partitions = newPartitions

    # 初始化上下界矩阵和子节点集合
    upper = [[float("inf") for _ in range(len(partitions))] for _ in range(k)]
    lower = [[float("-inf") for _ in range(len(partitions))] for _ in range(k)]
    children = []

    # 计算每个子集的上下界并递归构建子节点
    for i, par in enumerate(partitions):
        for j in range(k):
            if len(par) > 0:
                d = [distance_function(VP[j], p) for p in par]
                lower[j][i] = min(d)  # 计算下界
                upper[j][i] = max(d)  # 计算上界
        children.append(MVPTBulkload(par, MaxLeafSize, k, num_regions, distance_function))

    # 递归构建左子树和右子树，并返回内部节点
    return MVPTInternalNode(VP, children, lower, upper)

# MVP 树范围查询算法
def MVPTRangeSearch(node, query_point, radius, distance_function):
    """
    MVP 树的范围查询算法
    :param node: 当前查询的节点（VPTInternalNode 或 PivotTable）
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
    distance_VPs_q = []
    for pivot in node.pivots:
        distance_vp_q = distance_function(pivot, query_point)
        distance_VPs_q.append(distance_vp_q)
        if distance_vp_q <= radius:
            result.append(pivot)

    for i, child in enumerate(node.children):
        if not child:
            continue

        done = False
        for j, pivot in enumerate(node.pivots):
            if distance_VPs_q[j] + node.upperBound[j][i] <= radius:
                result.extend(getAllData(child))
                done = True
                break

            if distance_VPs_q[j] + radius <= node.lowerBound[j][i] or distance_VPs_q[j] - radius >= node.upperBound[j][i]:
                done = True
                break

        if done == False:
            result.extend(MVPTRangeSearch(child, query_point, radius, distance_function))

    return result

if __name__ == "__main__":
    # 参数
    num_data_points = 100  # 数据点数量
    dimensions = 2         # 数据点维度
    query_point = [25, 33]  # 查询点
    search_radius = 20      # 查询半径
    max_leaf_size = 5       # 叶子节点大小上界
    pivots_num = 2         # 支撑点数量
    num_regions = 3       # 支撑点划分的区域
    file_path = "./data/"   # 文件路径
    data_file_name = f"data_{num_data_points}points_{dimensions}dimensions.json"  # 数据点文件名

    # 距离函数
    minkowski_t = 2  # 闵可夫斯基距离的参数 t or float('inf')
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

    # 构建 MVP 树
    mvpt_root = MVPTBulkload(data, max_leaf_size, pivots_num, num_regions, minkowski_distance)

    # 打印部分数据
    print("Data points (first 5):", data[:5])
    print("Query point:", query_point)
    print("Search radius:", search_radius)
    print(f'MVPT.pivots: {mvpt_root.pivots}')
    print(f'len(MVPT.children): {len(mvpt_root.children)} == num_regions {num_regions} ^ pivots_num {pivots_num}')
    for i, child in enumerate(mvpt_root.children):
        for j, pivot in enumerate(mvpt_root.pivots):
            print(f'pivot {j + 1} children {i + 1} lowerBound: {mvpt_root.lowerBound[j][i]}, '
                  f'upperBound: {mvpt_root.upperBound[j][i]}')

    # 使用范围查询算法
    result = MVPTRangeSearch(mvpt_root, query_point, search_radius, minkowski_distance)

    # 输出查询结果
    print("Query result (points within radius):")
    print(result)
