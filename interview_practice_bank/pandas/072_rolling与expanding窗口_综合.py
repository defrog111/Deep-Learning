"""
题目 072：rolling与expanding窗口_综合

要求：完成“rolling与expanding窗口”的综合题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造时间顺序数值。
2. 计算移动窗口均值。
3. 计算从起点累计均值。
4. 计算指数加权均值。
5. 最终累计均值等于全局均值。
6. 汇总窗口结果。

完成标准：
- 最终累计均值等于全局均值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
series = pd.Series([2, 4, 6, 8, 10], dtype=float)  # 构造时间顺序数值。
rolling_mean = series.rolling(window=5, min_periods=1).mean()  # 计算移动窗口均值。
expanding_mean = series.expanding(min_periods=1).mean()  # 计算从起点累计均值。
ewm_mean = series.ewm(alpha=0.5, adjust=False).mean()  # 计算指数加权均值。
assert expanding_mean.iloc[-1] == series.mean()  # 最终累计均值等于全局均值。
print(pd.DataFrame({'value': series, 'rolling': rolling_mean, 'expanding': expanding_mean, 'ewm': ewm_mean}))  # 汇总窗口结果。
