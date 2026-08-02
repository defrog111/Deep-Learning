"""
题目 094：时间序列切分_变式

要求：完成“时间序列切分”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. gap或embargo避免标签窗口与未来特征相邻泄漏。
3. 验证时间间隔。

完成标准：
- 验证时间间隔。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
ordered_time = np.arange(20); gap = 2; train_time = ordered_time[:10]; validation_time = ordered_time[10 + gap:15]  # gap或embargo避免标签窗口与未来特征相邻泄漏。
assert train_time.max() + gap < validation_time.min()  # 验证时间间隔。
