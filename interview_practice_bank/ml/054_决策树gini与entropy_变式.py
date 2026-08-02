"""
题目 054：决策树Gini与Entropy_变式

要求：完成“决策树Gini与Entropy”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 计算entropy信息增益。
3. 验证有效切分不增加加权不纯度。

完成标准：
- 验证有效切分不增加加权不纯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
parent_counts = np.array([6, 4]); left_counts = np.array([4, 1]); right_counts = parent_counts - left_counts; entropy_fn = lambda counts: -(counts[counts > 0] / counts.sum() * np.log2(counts[counts > 0] / counts.sum())).sum(); information_gain = entropy_fn(parent_counts) - (left_counts.sum() * entropy_fn(left_counts) + right_counts.sum() * entropy_fn(right_counts)) / parent_counts.sum()  # 计算entropy信息增益。
assert information_gain >= 0  # 验证有效切分不增加加权不纯度。
