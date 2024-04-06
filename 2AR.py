import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
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
data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
plt.ylabel('流量')
plt.title("流量时间序列分析图")
plt.show()

# 绘制自相关图和差分后序列的自相关图
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
# 自相关图
plot_acf(data, ax=axes[0])
axes[0].set_title("原始序列的自相关图")
# 平稳性检测
print(u'原始序列的ADF检验结果为：', sm.tsa.stattools.adfuller(data['data1']))
# 差分处理

diff_data = data.diff().dropna()
# 差分后序列的自相关图
plot_acf(diff_data, ax=axes[1])
axes[1].set_title("差分后序列的自相关图")

plt.tight_layout()
plt.show()
# 定阶
warnings.filterwarnings(action="ignore")
model = AutoReg(data, lags=3)
result = model.fit()
# 输出模型摘要信息
print('模型报告为：\n', result.summary())
# 残差分析
residuals = result.resid

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

# 预测未来5天
forecast = result.predict(start=len(data), end=len(data)+4)
forecast = data['data1'].iloc[-1] + forecast.cumsum()
print('预测未来5天的结果为：\n', forecast)

# 绘制实际观测值和拟合值的曲线
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['data1'], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Forecast')
plt.legend()
plt.show()