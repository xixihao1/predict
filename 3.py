import numpy as np
import matplotlib.pyplot as plt
import math

# 设置泊松分布的参数
lambda_param = 3.5

# 计算泊松分布的概率质量函数
k_values = np.arange(0, 15)
pmf_values = np.exp(-lambda_param) * np.power(lambda_param, k_values) / np.array([math.factorial(k) for k in k_values])

# 绘制泊松分布的概率质量函数图像
plt.plot(k_values, pmf_values, 'ro-', label='Poisson PMF')

# 添加标签和标题
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title('Poisson Distribution')

# 显示图像
plt.show()