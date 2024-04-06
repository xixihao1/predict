import pandas as pd

# 读取数据集
data = pd.read_excel('isp2.xlsx')

# 将日期列转换为Datetime格式
data['date'] = pd.to_datetime(data['date'])

# 年份加一
data['date'] = data['date'] + pd.DateOffset(years=1)

# 导出到isp2.xlsx
data.to_excel('isp4.xlsx', index=False)