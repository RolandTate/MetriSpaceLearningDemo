from tools import pivotSelectionRand, minkowski_distance_factory
import numpy as np

def maxVarianceSelection(data, pivots_num, distance_function):
    """
    使用最大方差选择算法选择支撑点

    :param data: 数据点集合
    :param pivots_num: 需要选择的支撑点个数
    :param distance_function: 距离函数
    :return: pivots: 选择的支撑点列表
    """

    # # 随机选择第一个支撑点
    pivots = pivotSelectionRand(data, 1)

    # 迭代选择剩余的支撑点
    for _ in range(1, pivots_num):
        # 存储每个数据点到当前支撑点集合的距离
        distances = []
        for x in data:
            # 对每个点计算到当前所有支撑点的距离
            dist_to_pivots = [distance_function(x, pivot) for pivot in pivots]
            distances.append(dist_to_pivots)

        # 转换为矩阵形式，行表示数据点，列表示不同支撑点的距离
        distances = np.array(distances)

        # 对距离计算方差（沿列计算）
        variances = distances.var(axis=1)

        # 选择方差最大的点作为下一个支撑点
        max_dist_pivot_idx = np.argmax(variances)
        pivots.append(data[max_dist_pivot_idx])

    return pivots


def maxSeparatedSelection(data, pivots_num, distance_function):
    """
    使用最大分离度选择算法选择支撑点。

    :param data: 数据点集合
    :param pivots_num: 需要选择的支撑点个数
    :param distance_function: 距离函数
    :return: pivots: 选择的支撑点列表
    """
    # 随机选择第一个支撑点
    pivots = pivotSelectionRand(data, 1)

    # 迭代选择剩余的支撑点
    for _ in range(1, pivots_num):
        # 计算每个点与当前所有支撑点的距离和
        distance_sums = []
        for x in data:
            # 对每个点计算到当前所有支撑点的距离之和
            dist_to_pivots_sum = sum(distance_function(x, pivot) for pivot in pivots)
            distance_sums.append(dist_to_pivots_sum)

        # 选择距离和最大的点作为下一个支撑点
        max_dist_pivot_idx = np.argmax(distance_sums)
        pivots.append(data[max_dist_pivot_idx])

    return pivots


def farthestFirstTraversalSelection(data, pivots_num, distance_function):
    """
    使用 Farthest-First-Traversal 算法选择支撑点。

    :param data: 数据点集合
    :param pivots_num: 需要选择的支撑点个数
    :param distance_function: 距离函数
    :return: pivots: 选择的支撑点列表
    """
    # 随机选择第一个支撑点
    pivots = pivotSelectionRand(data, 1)

    # 迭代选择剩余的支撑点
    for _ in range(1, pivots_num):
        # 存储每个数据点到当前所有支撑点的最小距离
        min_distances = []
        for x in data:
            # 对每个点计算到所有支撑点的最小距离
            min_distance = min(distance_function(x, pivot) for pivot in pivots)
            min_distances.append(min_distance)

        # 选择最小距离最大的点作为下一个支撑点
        max_min_dist_idx = np.argmax(min_distances)
        pivots.append(data[max_min_dist_idx])

    return pivots


def maxMeanSelection(data, pivots_num, distance_function):
    """
    使用平均值最大算法选择支撑点。
    因为个数一定，所以没必要求平均。
    切比雪夫距离计算是基于支撑点空间而不是原始度量空间。

    :param data: 数据点集合
    :param pivots_num: 需要选择的支撑点个数
    :param distance_function: 距离函数
    :return: pivots: 选择的支撑点列表
    """
    # 随机选择第一个支撑点
    pivots = pivotSelectionRand(data, 1)
    Chebyshev_distance = minkowski_distance_factory(float('inf'))

    # 迭代选择剩余的支撑点
    for _ in range(1, pivots_num):
        # 存储每个候选点到当前支撑点的所有距离的切比雪夫距离总和
        distance_sums = []
        for x in data:
            # 对于每个点，计算其到所有已选支撑点的最大距离，因为个数一定，所以没有必要求平均
            dist_to_pivots_sum = sum(Chebyshev_distance(x, pivot) for pivot in pivots)
            distance_sums = [].append(dist_to_pivots_sum)

        # 选择使得切比雪夫距离总和最大的点作为下一个支撑点
        max_mean_dist_idx = np.argmax(distance_sums)
        pivots.append(data[max_mean_dist_idx])

    return pivots
