"""
Author:hi
time:2024
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# 加载数据
df = pd.read_excel('int.xlsx', index_col='date', parse_dates=True)
data = df['data1']

# 为了简化，这里直接进行一阶差分
data_diff = data.diff().dropna()  # 一阶差分并移除NA值

# 检查差分后时间序列的平稳性
result_diff = adfuller(data_diff)
print('ADF Statistic after differencing: %f' % result_diff[0])
print('p-value after differencing: %f' % result_diff[1])

# 根据需要调整这些参数
p, d, q = 1, 0, 1  # 由于已经手动进行了一阶差分，这里的d设置为0

# 应用ARIMA模型到差分后的数据
model_diff = ARIMA(data_diff, order=(p, d, q))
model_fit_diff = model_diff.fit()

# 预测
forecast_diff = model_fit_diff.forecast(steps=5)

# 将预测结果转换回原始比例（如果你的模型基于差分数据）
# 注意：这个转换过程取决于你的差分方式。这里只是一个基础示例
forecast_original_scale = data.iloc[-1] + np.cumsum(forecast_diff)

print(forecast_original_scale)
##模型评价：准
"""2005-06-19 19:00:00    1.911342e+09
2005-06-19 20:00:00    1.748604e+09
2005-06-19 21:00:00    1.640440e+09
2005-06-19 22:00:00    1.568004e+09
2005-06-19 23:00:00    1.518959e+09"""