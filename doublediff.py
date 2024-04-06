import warnings
import pandas as pd #开源的Python库，用于数据分析和数据处理
import matplotlib.pyplot as plt #用于创建各种类型的图表和图形
import statsmodels.api as sm  #statsmodels是一个用于进行统计建模和计量经济学分析的python库
from statsmodels.tsa.arima.model import ARIMA #arima模型
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # 自相关,偏自相关
from statsmodels.stats.diagnostic import acorr_ljungbox # 白噪声检验
from statsmodels.stats.stattools import durbin_watson # D-W检验,差分自相关检验
from statsmodels.graphics.gofplots import qqplot # QQ图,检验一组数据是否服从正态分布
from statsmodels.tsa.stattools import adfuller as ADF # 平稳性检测
warnings.filterwarnings(action="ignore")  # 忽略告警
# 参数初始化
discfile = 'isp2.xlsx'

# 读取数据，指定日期列为指标，pandas自动将“日期”列识别为Datetime格式
data = pd.read_excel(discfile) #读取Excel文件
print("打印数据的基本统计信息\n", data.describe())
print(data.info())
print("检查数据中是否存在重复行\n", data.duplicated(subset=['date', 'data1']))
data = data.drop_duplicates(subset=['date', 'data1'], keep='first')#删除重复行
data = data.dropna()#删除缺失值
print("再次打印数据的基本统计信息\n", data.describe())
print(data.info())
data = data.set_index('date') #将'date'列设置为数据的索引
print("打印数据的前几行\n", data.head())

# 时序图
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)#图表控制
plt.ylabel('流量')
plt.title("网络流量时间序列分析图")
plt.show()#显示绘制的图表

# 自相关图
plot_acf(data)
plt.title("原始序列的自相关图")
plt.show()

# 平稳性检测
print('原始序列的ADF检验结果为：', ADF(data[u'data1']))
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore
print('初始序列的白噪声检验结果为：', acorr_ljungbox(data, lags=1))  # 返回统计量和p值,lags=1参数指定了要计算的滞后阶数

# 差分后的结果
D1_data = data.diff().dropna()#使用`diff()`函数对数据进行一阶差分
D1_data.columns = ['流量差分']
D1_data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)  # 时序图
plt.title("一阶差分之后序列的时序图")
plt.ylabel('流量')
plt.show()

# 自相关图
plot_acf(D1_data)
plt.title("一阶差分之后序列的自相关图")
plt.show()
print('一阶差分序列的ADF检验结果为：', ADF(D1_data['流量差分']))  # 平稳性检测
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 白噪声检验
print('一阶差分序列的白噪声检验结果为：', acorr_ljungbox(D1_data, lags=1))  # 返回统计量和p值,lags=1参数指定了要计算的滞后阶数。

# 偏自相关图
plot_pacf(D1_data)
plt.title("一阶差分后序列的偏自相关图")
plt.show()

#二阶
# 差分后的结果
D_data = D1_data.diff().dropna()#再使用`diff()`函数对数据进行2阶差分
D_data.columns = ['流量差分']
D_data.plot(color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)  # 时序图
plt.title("二阶差分之后序列的时序图")
plt.ylabel('流量')
plt.show()

# 自相关图
plot_acf(D_data)
plt.title("二阶差分之后序列的自相关图")
plt.show()
print('二阶差分序列的ADF检验结果为：', ADF(D_data['流量差分']))  # 平稳性检测
# 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

# 白噪声检验
print('二阶差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值,lags=1参数指定了要计算的滞后阶数。

# 偏自相关图
plot_pacf(D_data)
plt.title("二阶差分后序列的偏自相关图")
plt.show()

# 定阶
data['data1'] = data['data1'].astype(float)
pmax = int(len(data) / 40)#一般阶数不超过length/10
qmax = int(len(data) / 40)#一般阶数不超过length/10
#确定了AR和MA的最大阶数。这里使用了经验法则，将数据长度除以10作为最大阶数。
bic_matrix = []#贝叶斯信息准则空列表
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):#使用嵌套循环遍历所有可能的AR和MA阶数的组合
        try:#尝试拟合ARIMA模型并计算BIC值，存在部分报错，所以用try来跳过报错
            tmp.append(ARIMA(data, order=(p, 1, q)).fit().bic)
        except:#如果出现异常（例如模型无法拟合），则将值设为`None`
            tmp.append(None)
    bic_matrix.append(tmp)#将每个阶数组合的BIC值添加到
bic_matrix = pd.DataFrame(bic_matrix)
print('BIC矩阵：')
print(bic_matrix)

tmp_data = bic_matrix.values#这行代码将BIC矩阵转换为一个NumPy数组，并赋值给`tmp_data`变量
tmp_data = tmp_data.flatten()#这行代码将`tmp_data`数组展平为一维数组
s = pd.DataFrame(tmp_data, columns=['value'])#这行代码将展平后的一维数组创建为一个DataFrame，并将列名设置为'value'。
s = s.dropna()#这行代码删除DataFrame中的缺失值（NaN）所在的行
print('BIC最小值：', s.min())#`min()`函数找到DataFrame中'value'列的最小值

p, q = bic_matrix.stack().idxmin()# 先用stack展平，然后用idxmin找出最小值位置
print('BIC最小的p值和q值为：%s、%s' % (p, q))

# 建立ARIMA模型
model = sm.tsa.ARIMA(data, order=(p, 1, q))#`p`表示自回归阶数，`1`表示差分阶数，`q`表示移动平均阶数
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

print('D-W检验的结果为：', durbin_watson(residuals.values))#使用Durbin-Watson检验来评估残差序列是否存在自相关。
print('残差序列的白噪声检验结果为：', acorr_ljungbox(residuals, lags=1))#使用Ljung-Box检验来评估残差序列是否为白噪声

# 预测未来5天
forecast = model_fit.forecast(steps=100)
print('预测未来5天：\n', forecast)

# 绘制实际观测值和拟合值的曲线
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['data1'], label='Actual')
plt.plot(forecast.index, forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs Forecast')
plt.legend()
plt.show()