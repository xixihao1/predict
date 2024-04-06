import warnings
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 数据可视化库
import statsmodels.api as sm  # 统计模型库

# 读取数据
df = pd.read_excel('isp2.xlsx', index_col='date')
# 用Pandas工具查看数据集的前几行
"""
print("查看数据集的前5行:")
print(df.head())

# 查看数据集摘要
print("查看数据集摘要:")
print(df.info())

# 数据描述性统计分析查看数据的平均值、标准差、最小值、分位数、最大值
print("数据描述性统计分析:")
print(df.describe())

# rate变量分布直方图;Matplotlib 工具的hist()方法绘制直方图：
fig = plt.figure(figsize=(8, 5))  # 设置画布大小
warnings.filterwarnings(action="ignore")
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
data_tmp = df['data1']  # 过滤出data变量的样本
# 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
plt.hist(data_tmp, bins='auto', color='g')
plt.xlabel('data1')  # 设置x轴名称
plt.ylabel('数量')  # 设置y轴名称
plt.title('data1变量分布直方图')  # 设置标题的名称
plt.show()  # 展示图片

# 绘制折线图
df.plot(title='Federal funds data', figsize=(12, 3))  # 绘图
plt.show()  # 展示图片
"""
# 构建季马尔可夫切换动态回归模型
mod_fedfunds = sm.tsa.MarkovRegression(df, k_regimes=2)  # 建模: k_regimes=2参数指定马尔可夫链的状态数
res_fedfunds = mod_fedfunds.fit()  # 拟合
print("输出模型摘要信息:")
print(res_fedfunds.summary())  # 输出模型摘要信息


# 绘制高状态概率图 .smoothed_marginal_probabilities[1]：.smoothed_marginal_probabilities 是模型提供的一个属性，它包含了每个观测点处于各个状态的概率估计值。在这里
# [1] 表示选择第二个状态的概率分布，即处于“高”流量状态的概率。
res_fedfunds.smoothed_marginal_probabilities[1].plot(
    title='Probability of being in the high regime', figsize=(12, 3))
plt.show()  # 展示图片

# 模型中不同状态之间的预期持续时间
# 列出了每个状态在不发生转换时预计将持续的平均时间（可以是观测单位如天数、月份或年份）。对于双模态模型（例如高利率和低利率两个状态），此属性将提供两个值，分别对应于处于“高”利率状态和“低”利率状态下的条件期望持续时间。
print('****res_fedfunds.expected_durations 不同状态之间的预期持续时间****')
print(res_fedfunds.expected_durations)
