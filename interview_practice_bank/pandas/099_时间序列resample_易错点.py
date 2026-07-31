"""
题目 099：时间序列resample_易错点

要求：完成“时间序列resample”的易错点题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
index = pd.date_range('2025-01-01', periods=10, freq='12h')  # 创建半日频率时间索引。
series = pd.Series(range(10), index=index, name='value')  # 构造时间序列。
daily = series.resample('D').agg(['sum', 'mean', 'count'])  # 降采样到每日并多重聚合。
shifted = series.shift(3, freq='h')  # 移动时间索引而不移动值。
assert daily['count'].sum() == len(series)  # 验证重采样未丢数据。
print(daily, '\n', shifted.head())  # 输出每日聚合和索引平移。
