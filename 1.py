import numpy as np
import pandas as pd


def frac_diff(series, d, window):
    """
    分数差分的简单实现。

    参数:
    series: pd.Series，时间序列数据。
    d: float，分数差分的阶数。
    window: int，用于计算权重的窗口大小。

    返回:
    fd_series: 分数差分后的时间序列。
    """
    # 初始化权重向量
    weights = [1]
    for k in range(1, window):
        weights.append(-weights[-1] * ((d - k + 1)) / k)
    weights = np.array(weights[::-1])

    # 创建分数差分序列
    fd_series = []
    for i in range(window, series.shape[0]):
        windowed_values = series.values[(i - window):i]
        fd_value = np.dot(weights, windowed_values)
        fd_series.append(fd_value)

    # 转换为pandas Series并保持时间索引的对应性
    fd_series = pd.Series(fd_series, index=series.index[window:])

    return fd_series


# 示例用法
if __name__ == "__main__":
    # 导入数据集
    df = pd.read_excel("int.xlsx")

    # 构建时间序列数据
    series = pd.Series(df['data1'].values, index=df['date'])

    # 应用分数差分，假设d=0.5，窗口大小为10
    fd_series = frac_diff(series, 0.5, 10)

    # 将分数差分结果保存到int1.xlsx文件
    fd_series.to_excel("int1.xlsx")