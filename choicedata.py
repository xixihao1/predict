"""
Author:hi
time:2024
"""
import warnings
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 数据可视化库
import statsmodels.api as sm  # 统计模型库

# 读取数据
df = pd.read_excel('int.xlsx', index_col='date')
# 用Pandas工具查看数据集的前几行
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
df.plot(title='network traffic dataset', figsize=(12, 3), color='k')  # 绘图
plt.show()  # 展示图片