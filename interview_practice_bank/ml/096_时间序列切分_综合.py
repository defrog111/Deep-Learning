"""
题目 096：时间序列切分_综合

要求：完成“时间序列切分”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合比较扩展窗口与固定滚动窗口。
3. 验证两种回测。

完成标准：
- 验证两种回测。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
time_values = np.arange(12); expanding_splits = [(time_values[:end], time_values[end:end + 2]) for end in range(4, 11, 2)]; rolling_splits = [(time_values[max(0, end - 4):end], time_values[end:end + 2]) for end in range(4, 11, 2)]  # 综合比较扩展窗口与固定滚动窗口。
assert len(expanding_splits[-1][0]) > len(rolling_splits[-1][0]) and all(train.max() < test.min() for train, test in rolling_splits)  # 验证两种回测。
