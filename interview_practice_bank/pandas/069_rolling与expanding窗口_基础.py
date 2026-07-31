"""
题目 069：rolling与expanding窗口_基础

要求：完成“rolling与expanding窗口”的基础题，并解释输出的 shape、索引和数据类型。
先自己实现，再运行本文件查看参考代码结果。
"""

import pandas as pd  # 导入 Pandas。
series = pd.Series([2, 4, 6, 8, 10], dtype=float)  # 构造时间顺序数值。
rolling_mean = series.rolling(window=2, min_periods=1).mean()  # 计算移动窗口均值。
expanding_mean = series.expanding(min_periods=1).mean()  # 计算从起点累计均值。
ewm_mean = series.ewm(alpha=0.5, adjust=False).mean()  # 计算指数加权均值。
assert expanding_mean.iloc[-1] == series.mean()  # 最终累计均值等于全局均值。
print(pd.DataFrame({'value': series, 'rolling': rolling_mean, 'expanding': expanding_mean, 'ewm': ewm_mean}))  # 汇总窗口结果。
