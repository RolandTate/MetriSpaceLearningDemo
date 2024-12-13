import random
import math

# 欧氏距离函数
def euclidean_distance(p1, p2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(p1, p2)))

# PivotTable 类 (叶子节点)
class PivotTable:
    def __init__(self, data):
        self.data = data  # 存储叶子节点的数据点

# 广义 MVP 树节点类
class MVPTInternalNode:
    def __init__(self, vantage_points, partitions, cutoff_values, children):
        self.vantage_points = vantage_points  # 支撑点集合
        self.partitions = partitions  # 每个支撑点划分的区域
        self.cutoff_values = cutoff_values  # 每个支撑点的划分半径
        self.children = children  # 子节点

# 随机选择支撑点
def select_vantage_points(data, num_points):
    return random.sample(data, num_points)

# 按支撑点划分数据集
def partition_data(data, vantage_point, num_regions):
    distances = [(point, euclidean_distance(point, vantage_point)) for point in data]
    distances.sort(key=lambda x: x[1])  # 按距离排序
    partition_size = len(data) // num_regions
    partitions = []
    for i in range(num_regions):
        start = i * partition_size
        end = start + partition_size if i < num_regions - 1 else len(data)
        partitions.append([x[0] for x in distances[start:end]])  # 划分数据
    cutoff_values = [distances[i * partition_size][1] for i in range(1, num_regions)]
    return partitions, cutoff_values

# 构建广义 MVP 树
def GMVPTree(data, m, v, k, p, level=1):
    """
    构建广义 MVP 树
    :param data: 数据点集合
    :param m: 每个支撑点的划分区域数量
    :param v: 每个节点的支撑点数量
    :param k: 叶子节点的最小数据量
    :param p: 最大允许的树层数
    :param level: 当前层数
    :return: 广义 MVP 树的根节点
    """
    # 如果数据集为空，返回空树
    if len(data) == 0:
        return None

    # 如果数据集规模小于等于 k，创建叶子节点
    if len(data) <= k:
        return PivotTable(data)

    # 选择支撑点
    vantage_points = select_vantage_points(data, v)
    data = [x for x in data if x not in vantage_points]  # 移除支撑点

    # 初始化分区与子节点集合
    partitions = [data]
    cutoff_values_all = []
    children = []

    # 遍历每个支撑点进行多次划分
    for i, vp in enumerate(vantage_points):
        new_partitions = []
        cutoff_values = []
        for region in partitions:
            if len(region) > 0:
                sub_partitions, sub_cutoff_values = partition_data(region, vp, m)
                new_partitions.extend(sub_partitions)
                cutoff_values.extend(sub_cutoff_values)
        partitions = new_partitions
        cutoff_values_all.append(cutoff_values)

    # 递归构建子树
    for partition in partitions:
        if len(partition) > 0:
            child = GMVPTree(partition, m, v, k, p, level + 1)
            children.append(child)

    # 创建内部节点
    return MVPTInternalNode(vantage_points, partitions, cutoff_values_all, children)

# 随机生成数据
def generate_data(num_points, dimensions, lower=0, upper=100):
    return [[random.randint(lower, upper) for _ in range(dimensions)] for _ in range(num_points)]

# 主程序
if __name__ == "__main__":
    # 参数
    num_data_points = 50  # 数据点数量
    dimensions = 2  # 数据点维度
    m = 3  # 每个支撑点划分的区域数量
    v = 2  # 每个节点的支撑点数量
    k = 5  # 叶子节点的最大数据量
    p = 10  # 最大树深度

    # 生成随机数据
    data = generate_data(num_data_points, dimensions)
    print("Generated data:", data[:5])  # 只打印部分数据

    # 构建广义 MVP 树
    gmvptree_root = GMVPTree(data, m, v, k, p)
    print("Generalized MVP Tree constructed.")
