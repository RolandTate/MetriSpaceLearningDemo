from tools import pivotSelectionRand
import numpy as np

def maxVarianceSelection(data, pivots_num, distance_function):
    """
        使用最大方差选择算法选择支撑点
        :param data: 数据点集合
        :param pivots_num: 需要选择的支撑点个数
        :param distance_function: 距离函数
        :return: pivots: 选择的支撑点列表
    """

    pivots = []

    # 随机选择第一个支撑点
    pivots.append(pivotSelectionRand(data, 1)[0])
    # 手动移除支撑点 VP
    data = [x for x in data if x not in pivots]

    # 迭代选择剩余的支撑点
    for _ in range(1, pivots_num):
        # 计算每个数据点到当前支撑点集合的距离
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
