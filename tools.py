import random
import os
import json


# 数据生成函数
def generate_data(num_points, dimensions, lower=0, upper=100, save_to_file=False, file_name="data.json",
                  file_path="./"):
    """
    生成随机整数数据，并可选存储到文件；如果文件已存在，则直接加载
    :param num_points: 数据点数量
    :param dimensions: 数据点维度
    :param lower: 生成数据的最小值
    :param upper: 生成数据的最大值
    :param save_to_file: 是否将生成的数据存储到文件
    :param file_name: 文件名
    :param file_path: 文件路径
    :return: 生成的数据或从文件加载的数据
    """
    full_path = os.path.join(file_path, file_name)

    # 检查文件是否已经存在
    if os.path.exists(full_path):
        print(f"File already exists: {full_path}. Loading data from file.")
        return read_data(file_name=file_name, file_path=file_path)

    # 生成新数据
    data = [[random.randint(lower, upper) for _ in range(dimensions)] for _ in range(num_points)]

    # 如果需要保存到文件
    if save_to_file:
        os.makedirs(file_path, exist_ok=True)  # 确保路径存在
        with open(full_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Data saved to {full_path}")

    return data


# 数据读取函数
def read_data(file_name="data.json", file_path="./"):
    """
    从文件中读取数据
    :param file_name: 文件名
    :param file_path: 文件路径
    :return: 从文件中读取的数据
    """
    full_path = os.path.join(file_path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found: {full_path}")

    with open(full_path, "r") as file:
        data = json.load(file)

    print(f"Data loaded from {full_path}")
    return data

def minkowski_distance_factory(t):
    if t == float('inf'):  # 检查 t 是否为无穷
        def minkowski_distance(p1, p2):
            return max(abs(x - y) for x, y in zip(p1, p2))  # 切比雪夫距离
    else:
        def minkowski_distance(p1, p2):
            return sum(abs(x - y) ** t for x, y in zip(p1, p2)) ** (1 / t)
    return minkowski_distance

def pivotSelectionRand(data, pivots_num):
    return random.sample(data, pivots_num)

def determineSplitRadius(data, pivot, distanc_function):
    distances = [distanc_function(pivot, point) for point in data]
    distances.sort()  # 排序所有距离

    # 计算中位数
    n = len(distances)
    if n % 2 == 1:
        # 奇数取中间
        return distances[n // 2]
    else:
        # 偶数取平均
        return (distances[n // 2 - 1] + distances[n // 2]) / 2

def getAllData(node):
    from Pivot_Table import PivotTable
    from VantagePointTree import VPTInternalNode
    from MultipleVantagePointTree import MVPTInternalNode

    if isinstance(node, PivotTable):
        return node.data  # 叶子节点返回所有数据点

    result = []

    # 如果是 VP 树
    if isinstance(node, VPTInternalNode):
        result.append(node.pivot)  # 添加支撑点
        result.extend(getAllData(node.left))  # 获取左子树数据
        result.extend(getAllData(node.right))  # 获取右子树数据

    # 如果是 MVP 树
    if isinstance(node, MVPTInternalNode):
        result.extend(node.pivots)  # 添加支撑点
        result.extend(getAllData(node.children))  # 获取所有分区

    return result

