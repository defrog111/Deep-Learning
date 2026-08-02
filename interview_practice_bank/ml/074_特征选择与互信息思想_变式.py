"""
题目 074：特征选择与互信息思想_变式

要求：完成“特征选择与互信息思想”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 过滤法可先删除高度共线特征。
3. 验证冗余特征识别。

完成标准：
- 验证冗余特征识别。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
feature_matrix = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0], [4, 4, 1]], dtype=float); feature_correlation = np.corrcoef(feature_matrix, rowvar=False); redundant_pair = abs(feature_correlation[0, 1]) > 0.95  # 过滤法可先删除高度共线特征。
assert redundant_pair  # 验证冗余特征识别。
