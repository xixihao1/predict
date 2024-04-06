"""
Author:hi
time:2024
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.gofplots import qqplot
import statsmodels.api as sm

# 参数初始化
discfile = 'isp2.xlsx'
forecastnum = 5

# 读取数据，指定日期列为指标，pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile)
data = data.drop_duplicates(subset=['date', 'data1'], keep='first')
data = data.dropna()
data = data.set_index('date')

# 时序图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 平稳性检测
print(u'原始序列的ADF检验结果为：', sm.tsa.stattools.adfuller(data['data1']))

# 建立MA模型
model = sm.tsa.MarkovRegression(data, k_regimes=2)
# 拟合模型
model_fit = model.fit()

# 输出模型摘要信息
print('模型报告为：\n', model_fit.summary())

# 残差分析
residuals = model_fit.resid
# 绘制残差自相关图、残差偏自相关图和残差Q-Q图
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# 残差自相关图
plot_acf(residuals, ax=axes[0])
axes[0].set_title("残差自相关图")

# 残差偏自相关图
plot_pacf(residuals, ax=axes[1])
axes[1].set_title("残差偏自相关图")

# 残差Q-Q图
qqplot(residuals, line='q', fit=True, ax=axes[2])
axes[2].set_title("残差Q-Q图")

plt.tight_layout()
plt.show()

print('D-W检验的结果为：', durbin_watson(residuals.values))
print('残差序列的白噪声检验结果为：', acorr_ljungbox(residuals, lags=1))

# 绘制高状态概率图 .smoothed_marginal_probabilities[1]：.smoothed_marginal_probabilities 是模型提供的一个属性，它包含了每个观测点处于各个状态的概率估计值。在这里
# [1] 表示选择第二个状态的概率分布，即处于“高”流量状态的概率。
model_fit.smoothed_marginal_probabilities[1].plot(
    title='Probability of being in the high regime', figsize=(12, 3))
plt.show()  # 展示图片

# 模型中不同状态之间的预期持续时间
# 列出了每个状态在不发生转换时预计将持续的平均时间（可以是观测单位如天数、月份或年份）。对于双模态模型（例如高利率和低利率两个状态），此属性将提供两个值，分别对应于处于“高”利率状态和“低”利率状态下的条件期望持续时间。
print('****res_fedfunds.expected_durations 不同状态之间的预期持续时间****')
print(model_fit.expected_durations)