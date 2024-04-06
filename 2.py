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
#statsmodels库的ARMA模型在最新版本（0.13.0）中已经被弃用，并且从statsmodels.tsa.arima.model模块中移除。
#在statsmodels库的最新版本中，ARMA模型已经被ARIMA模型取代，因此ARMA模型的类没有直接提供。可以通过设置ARIMA模型的参数来实现ARMA模型。
#ARMA模型是ARIMA模型的一个特例，即差分阶数(d)为0。通过将ARIMA模型的差分阶数设置为0，就可以获得ARMA模型。
# 参数初始化
discfile = 'int1.xlsx'
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

# 自相关图
plot_acf(data)
plt.title("原始序列的自相关图")
plt.show()

# 平稳性检测
print(u'原始序列的ADF检验结果为：', sm.tsa.stattools.adfuller(data[u'data1']))

# 定阶
warnings.filterwarnings(action="ignore")
pmax = int(len(data) / 40)
qmax = int(len(data) / 40)
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(ARIMA(data, order=(p, 0, q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix)
print('BIC矩阵：')
print(bic_matrix)

tmp_data = bic_matrix.values
tmp_data = tmp_data.flatten()
s = pd.DataFrame(tmp_data, columns=['value'])
s = s.dropna()
print('BIC最小值：', s.min())

p, q = bic_matrix.stack().idxmin()
print('BIC最小的p值和q值为：%s、%s' % (p, q))

# 建立ARMA模型
model = sm.tsa.ARIMA(data, order=(p, 0, q))
# 拟合模型
model_fit = model.fit()

# 输出模型摘要信息
print('模型报告为：\n', model_fit.summary())

# 残差分析
residuals = model_fit.resid

plot_acf(residuals)
plt.title("残差自相关图")
plt.show()

plot_pacf(residuals)
plt.title("残差偏自相关图")
plt.show()

qqplot(residuals, line='q', fit=True)
plt.title("残差Q-Q图")
plt.show()

print('D-W检验的结果为：', durbin_watson(residuals.values))
print('残差序列的白噪声检验结果为：', acorr_ljungbox(residuals, lags=1))

# 预测未来5天
forecast = model_fit.forecast(steps=5)
print('预测未来5天，其预测结果、标准误差、置信区间如下：\n', forecast)