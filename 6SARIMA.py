import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf  # 自相关
from statsmodels.tsa.stattools import adfuller as ADF  # 平稳性检测
from statsmodels.graphics.tsaplots import plot_pacf  # 偏自相关
from statsmodels.stats.diagnostic import acorr_ljungbox  # 白噪声检验
import statsmodels.api as sm  # D-W检验,差分自相关检验
from statsmodels.graphics.api import qqplot  # QQ图,检验一组数据是否服从正态分布
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

# 读取数据
data = pd.read_excel('isp2.xlsx')
print('****************查看数据前5行********************')
print(data.head())
print('****************数据缺失查看********************')
print(data.info())
print('****************数据描述性统计分析********************')
print(data.describe())

data = data.set_index('date')

# 时序图

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
plt.ylabel('data1')
plt.title("时间序列分析图")
plt.show()

# 自相关图

plot_acf(data)
plt.title("原始序列的自相关图")
plt.show()

# 平稳性检测

print(u'原始序列的ADF检验结果为：', ADF(data[u'data1']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = [u'差分']
D_data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)  # 时序图
plt.title("一阶差分之后序列的时序图")
plt.ylabel('x1')
plt.show()
plot_acf(D_data)  # 自相关图
plt.title("一阶差分之后序列的自相关图")
plt.show()

print(u'差分序列的ADF检验结果为：', ADF(D_data[u'差分']))  # 平稳性检测

# 白噪声检验

print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值

plot_pacf(D_data)  # 偏自相关图
plt.title("一阶差分后序列的偏自相关图")
plt.show()

from itertools import product

# SARIMA的参数
ps = range(0, 1)
d = range(0, 2)
qs = range(0, 12)
# 季节项相关的参数
Ps = range(0, 1)
D = range(1, 2)
Qs = range(1, 2)
# 将参数打包，传入下面的数据，是那个BIC准则进行参数选择
params_list = list(product(ps, d, qs, Ps, D, Qs))
print('********************params_list***********************')
print(params_list)

from tqdm import tqdm


# 找最优的参数 SARIMAX
def find_best_params(data: np.array, params_list):
    result = []
    best_bic = 100000
    for param in tqdm(params_list):
        # 模型拟合
        model = SARIMAX(data, order=(param[0], param[1], param[2]),
                        seasonal_order=(param[3], param[4], param[5], 12)).fit(disp=-1)
        bicc = model.bic  # 拟合出模型的BIC值
        # print(bic)
        # 寻找最优的参数
        if bicc < best_bic:
            best_mode = model
            best_bic = bicc
            best_param = param
        param_1 = (param[0], param[1], param[2])
        param_2 = (param[3], param[4], param[5], 12)
        param = 'SARIMA{0}x{1}'.format(param_1, param_2)
        print(param)
        result.append([param, model.bic])

    result_table = pd.DataFrame(result)
    result_table.columns = ['parameters', 'bic']
    result_table = result_table.sort_values(by='bic', ascending=True).reset_index(drop=True)
    return result_table


#result_table = find_best_params(data, params_list)
#print(result_table)

model = SARIMAX(data, order=(0, 1, 6), seasonal_order=(0, 1, 1, 12)).fit()  # 建立SARIMAX模型
print('模型报告为：\n', model.summary())
resid = model.resid
# 自相关图
plot_acf(resid)
plt.title("残差自相关图")
plt.show()
# 偏自相关图
plot_pacf(resid)
plt.title("残差偏自相关图")
plt.show()
# 线性即正态分布
qqplot(resid, line='q', fit=True)
plt.title("残差Q-Q图")
plt.show()
# 解读：残差服从正态分布，均值为零，方差为常数
print('D-W检验的结果为：', sm.stats.durbin_watson(resid.values))
print('残差序列的白噪声检验结果为：', acorr_ljungbox(resid, lags=1))  # 返回统计量、P值

forecast = model.forecast(steps=5)
print('预测未来5天：\n', forecast)

# 绘制实际观测值和拟合值的曲线
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['data1'], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('SARIMA Actual vs Forecast')
plt.legend()
plt.show()