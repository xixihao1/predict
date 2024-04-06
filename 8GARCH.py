import pandas as pd
import numpy as np
import warnings
from arch import arch_model
from scipy.stats import shapiro
from scipy.stats import probplot
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib import pyplot as plt
warnings.filterwarnings(action="ignore")

plt.style.use('fivethirtyeight')
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# 读取数据
df = pd.read_excel('int.xlsx', index_col='date')
# 用Pandas工具查看前五行数据
print(df.head())
# 查看数据集摘要
print(df.info())

df.dropna(inplace=True)
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 网络流量趋势分析
df['data1'].plot(figsize=(10, 5), title=f'网络流量趋势分析')
plt.show()
# 网络流量波动自相关图
acf = plot_acf(df['data1'], lags=30, title=f'网络流量波动自相关图')
plt.show()
# 网络流量波动偏自相关图
pacf = plot_pacf(df['data1'], lags=30, title=f'网络流量波动偏自相关图')
plt.show()
# 网络流量白噪声检验
ljung_res = acorr_ljungbox(df['data1'], lags=40, boxpierce=True)
print(u'白噪声检验结果为：', ljung_res)

def ts_plot(residuals, stan_residuals, lags=40):
    residuals.plot(title='GARCH 模型残差图', figsize=(15, 10))
    plt.show()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax[0].set_title('GARCH 标准残差密度图')
    ax[1].set_title('GARCH 标准残差概率图')
    residuals.plot(kind='kde', ax=ax[0])
    probplot(stan_residuals, dist='norm', plot=ax[1])
    plt.show()
    plot_acf(stan_residuals, lags=lags, title='GARCH 标准残差自相关性图')
    plt.show()
    plot_pacf(stan_residuals, lags=lags, title='GARCH 标准残差偏自相关性图')
    plt.show()

# 建立
garch = arch_model(df['data1'], vol='GARCH', p=1, q=1, dist='normal')
fgarch = garch.fit(disp='off')
resid = fgarch.resid#计算了拟合后的残差
st_resid = np.divide(resid, fgarch.conditional_volatility)#标准化残差
ts_plot(resid, st_resid)#绘图
print('****************模型结果输出*********************')
print(fgarch.summary())

arch_test = het_arch(resid, nlags=50)
shapiro_test = shapiro(st_resid)

print(f'Lagrange mulitplier p-value: {arch_test[1]}')
print(f'F test p-value: {arch_test[3]}')
print(f'Shapiro-Wilks p-value: {shapiro_test[1]}')

# 模型优化：使用具有大范围p和q的网格搜索来找到最适合波动性的模型
def gridsearch(data, p_rng, q_rng):#网格搜索函数`gridsearch()`，用于寻找最适合波动性的GARCH模型
    top_score, top_results = float('inf'), None
    top_models = []
    for p in range(len(p_rng)):
        for q in range(len(q_rng)):
            model = arch_model(data, vol='GARCH', p=p_rng[p], q=q_rng[q], dist='normal')
            model_fit = model.fit(disp='off')
            resid = model_fit.resid
            st_resid = np.divide(resid, model_fit.conditional_volatility)
            results = evaluate_model(resid, st_resid)
            results['AIC'] = model_fit.aic#AIC是一种模型选择准则，用于衡量模型的拟合优度和复杂度
            results['params']['p'] = p
            results['params']['q'] = q
            if results['AIC'] < top_score:#条件判断语句会检查当前模型的AIC值是否比之前最佳模型的AIC值更低
                top_score = results['AIC']
                top_results = results

            elif results['LM_pvalue'][1] is False:
                top_models.append(results)

    top_models.append(top_results)
    return top_models

# 模型评估
def evaluate_model(residuals, st_residuals, lags=50):#残差序列）、（标准化残差序列）和`lags`
    results = {#存储评估结果
        'LM_pvalue': None,
        'F_pvalue': None,
        'SW_pvalue': None,
        'AIC': None,
        'params': {'p': None, 'q': None}
    }

    arch_test = het_arch(residuals, nlags=lags)
    shap_test = shapiro(st_residuals)

    results['LM_pvalue'] = [arch_test[1], arch_test[1] < 0.05]
    results['F_pvalue'] = [arch_test[3], arch_test[3] < 0.05]
    results['SW_pvalue'] = [shap_test[1], shap_test[1] < 0.05]

    return results

p_rng = list(range(1, 3))  # 为了缩短运行时间取3
q_rng = list(range(1, 5)) # 为了缩短运行时间取5
df['dif_pct_change'] = df['data1'].diff()  # 一阶差分 序列变为平稳非白噪声序列
df = df.reset_index()
top_models = gridsearch(df['dif_pct_change'].iloc[1:, ], p_rng, q_rng)
print('*****************优化模型*******************')
print(top_models)

# 建模：使用最优的 p  q 值
garch = arch_model(df['data1'], vol='GARCH', p=17, q=25, dist='normal')
fgarch = garch.fit(disp='off')
resid = fgarch.resid
st_resid = np.divide(resid, fgarch.conditional_volatility)
ts_plot(resid, st_resid)
arch_test = het_arch(resid, nlags=50)
shapiro_test = shapiro(st_resid)
print(f'Lagrange mulitplier p-value: {arch_test[1]}')
print(f'F test p-value: {arch_test[3]}')
print(f'Shapiro-Wilks p-value: {shapiro_test[1]}')
print('*****************fgarch.summary()*******************')
print(fgarch.summary())

print('预测未来5个单位，其预测方差和预测值：\n')
forecasts=fgarch.forecast(horizon=5)
print(forecasts.variance.dropna().head())
print(forecasts.mean.dropna().head())
