import pandas as pd

# 读取数据集
data = pd.read_excel('isp2.xlsx')

# 将日期列转换为Datetime格式
data['date'] = pd.to_datetime(data['date'])


# 导出到isp2.xlsx
data.to_excel('isp4.xlsx', index=False)
