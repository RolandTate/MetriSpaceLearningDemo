from tools import pivotSelectionRand, minkowski_distance_factory
import numpy as np
from tqdm import tqdm

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
        for x in [point for point in data if point not in pivots]:
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
        for x in [point for point in data if point not in pivots]:
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
        for x in [point for point in data if point not in pivots]:
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
        max_sum = 0
        best_pivot = None

        # 遍历剩余的每个点，假设其为新增的支撑点
        for candidate in tqdm(data):
            if candidate in pivots:
                continue

            # 假设 candidate 为新的支撑点，计算所有点在支撑点空间的投影
            candidate_pivots = pivots + [candidate]
            data_in_pivot_space = [
                [distance_function(x, pivot) for pivot in candidate_pivots] for x in data
            ]

            # 计算切比雪夫距离的总和
            total_chebyshev_sum = sum(
                Chebyshev_distance(x_projection, y_projection)
                for x_projection in data_in_pivot_space
                for y_projection in data_in_pivot_space
            )

            # 更新最大值
            if total_chebyshev_sum > max_sum:
                max_sum = total_chebyshev_sum
                best_pivot = candidate

        # 将最优的支撑点加入结果
        pivots.append(best_pivot)

    return pivots


def radius_sensitive_evaluation(evaluation_set, distance_function, pivot_set, r):
    """
    半径敏感的目标函数，计算在支撑点空间中，满足切比雪夫距离大于等于 r 的点对数量。

    :param evaluation_set: 用于评价的点集合
    :param distance_function: 距离函数
    :param pivot_set: 当前的支撑点集合
    :param r: 半径阈值
    :return: 满足条件的点对数量
    """
    Chebyshev_distance = minkowski_distance_factory(float('inf'))

    # 投影到支撑点空间
    projected_points = [
        [distance_function(x, pivot) for pivot in pivot_set] for x in evaluation_set
    ]

    # 计算满足切比雪夫距离大于等于 r 的点对数量
    count = 0
    for i, x_projection in enumerate(projected_points):
        for j, y_projection in enumerate(projected_points):
            if i != j:  # 避免点对自身的计算
                chebyshev_distance = Chebyshev_distance(x_projection, y_projection)
                if chebyshev_distance >= r:
                    count += 1
    return count


def incrementalSamplingPivotSelection(data, num_pivots, distance_function):
    """
    增量采样支撑点选择算法。

    :param data: 数据点集合
    :param distance_function: 距离函数
    :param num_pivots: 需要选择的支撑点个数
    :param candidate_function: 生成候选集合的函数
    :param evaluation_function: 评价函数，用于评估候选点在给定支撑点集合中的表现
    :return: 支撑点集合
    """
    # 初始化候选集合
    # candidate_set = candidate_function(data)
    # candidate_set = pivotSelectionRand(data, 5)
    candidate_set = farthestFirstTraversalSelection(data, 10, distance_function)

    # 初始化评估集合
    # evaluation_set = candidate_function(data)
    evaluation_set = pivotSelectionRand(data, 100)

    # 初始化支撑点集合
    pivots = []

    # 迭代选择支撑点
    for _ in range(num_pivots):
        best_value = 0  # 初始化最佳值为0
        best_point = None  # 初始化最佳点

        # 遍历候选集合，评估每个候选点
        for candidate in candidate_set:
            # 如果候选点已被选中过，则跳过
            if candidate in pivots:
                continue

            # 假设当前候选点加入支撑点集合
            current_pivot_set = pivots + [candidate]

            # 计算候选点的评价值
            # value = evaluation_function(evaluation_set, distance_function, current_pivot_set)
            value = radius_sensitive_evaluation(evaluation_set, distance_function, current_pivot_set, 200)

            # 更新最佳点
            if value > best_value:
                best_value = value
                best_point = candidate

        # 将最佳点加入支撑点集合
        pivots.append(best_point)

    return pivots
