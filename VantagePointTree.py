from tools import generate_data, minkowski_distance_factory, pivotSelection, determineSplitRadius, getAllData
from Pivot_Table import PivotTable, PTRangeSearch

# VP 树内部节点类
class VPTInternalNode:
    def __init__(self, pivot, splitRadius, left, right):
        self.pivot = pivot  # 支撑点
        self.splitRadius = splitRadius  # 划分半径
        self.left = left  # 左子树
        self.right = right  # 右子树

# VP 树批建算法
def VPTBulkload(data, MaxLeafSize, distance_function):
    """
    VP 树批建算法
    :param data: 数据点集合
    :param MaxLeafSize: 叶子节点大小上界
    :param distance_function: 距离函数
    :return: VP 树根节点
    """
    if 0 == len(data):
        return None
    # 当数据量小于等于 MaxLeafSize 时，构建 Pivot Table 作为叶子节点
    if len(data) <= MaxLeafSize:
        pivot = pivotSelection(data, 1)
        # 手动移除支撑点 VP
        data = [x for x in data if x not in pivot]
        return PivotTable(data, pivot, distance_function)  # 构建 PivotTable

    # 选择支撑点
    VP = pivotSelection(data, 1)[0]

    # 手动移除支撑点 VP
    data = [x for x in data if x != VP]

    # 确定划分半径
    R = determineSplitRadius(data, VP, distance_function)

    # 初始化左右子数据集
    leftData = []
    rightData = []

    # 按支撑点距离划分数据
    for s in data:
        if distance_function(s, VP) <= R:
            leftData.append(s)
        else:
            rightData.append(s)

    # 处理空子树
    left = VPTBulkload(leftData, MaxLeafSize, distance_function)
    right = VPTBulkload(rightData, MaxLeafSize, distance_function)

    # 递归构建左子树和右子树，并返回内部节点
    return VPTInternalNode(VP, R, left, right)

# VP 树范围查询算法
def VPTRangeSearch(node, query_point, radius, distance_function):
    """
    VP 树的范围查询算法
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
    distance_VP_q = distance_function(node.pivot, query_point)

    # 支撑点是查询结果
    if distance_VP_q <= radius:
        result.append(node.pivot)

    # 球内数据全部是查询结果
    if distance_VP_q + node.splitRadius <= radius:
        if node.left:
            result.extend(getAllData(node.left))

    # 球内侧不能排除
    elif distance_VP_q <= node.splitRadius + radius:
        if node.left:
            result.extend(VPTRangeSearch(node.left, query_point, radius, distance_function))

    # 球外侧不能排除
    if distance_VP_q + radius > node.splitRadius:
        if node.right:
            result.extend(VPTRangeSearch(node.right, query_point, radius, distance_function))

    return result

if __name__ == "__main__":
    # 参数
    num_data_points = 100  # 数据点数量
    dimensions = 2         # 数据点维度
    query_point = [50, 50]  # 查询点
    search_radius = 20      # 查询半径
    max_leaf_size = 5       # 叶子节点大小上界
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

    # 构建 VP 树
    vpt_root = VPTBulkload(data, max_leaf_size, minkowski_distance)

    # 打印部分数据
    print("Data points (first 5):", data[:5])
    print("Query point:", query_point)
    print("Search radius:", search_radius)
    print(f'VPT.pivot: {vpt_root.pivot}')

    # 使用范围查询算法
    result = VPTRangeSearch(vpt_root, query_point, search_radius, minkowski_distance)

    # 输出查询结果
    print("Query result (points within radius):")
    print(result)
