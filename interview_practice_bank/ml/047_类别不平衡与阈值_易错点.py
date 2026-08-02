"""
题目 047：类别不平衡与阈值_易错点

要求：完成“类别不平衡与阈值”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 阈值应依据业务成本而非固定0.5。
3. 高漏报成本下选择召回更高的方案。

完成标准：
- 高漏报成本下选择召回更高的方案。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
costs = {'fp': 1, 'fn': 10}; candidate_confusions = [np.array([[90, 5], [4, 1]]), np.array([[80, 15], [1, 4]])]; business_costs = [matrix[0, 1] * costs['fp'] + matrix[1, 0] * costs['fn'] for matrix in candidate_confusions]  # 阈值应依据业务成本而非固定0.5。
assert np.argmin(business_costs) == 1  # 高漏报成本下选择召回更高的方案。
