"""
题目 036：交叉验证_综合

要求：完成“交叉验证”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 外层评估泛化，内层选择超参数。
for outer_validation in outer_folds:  # 遍历外层验证块。
    outer_training = np.setdiff1d(np.arange(8), outer_validation); inner_validation = outer_training[::2]; inner_training = np.setdiff1d(outer_training, inner_validation); nested_counts.append((len(inner_training), len(inner_validation), len(outer_validation)))  # 只在外层训练集内部再切分。
3. 综合验证nested CV层级。

完成标准：
- 综合验证nested CV层级。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
outer_folds = [np.arange(0, 4), np.arange(4, 8)]; nested_counts = []  # 外层评估泛化，内层选择超参数。
for outer_validation in outer_folds:  # 遍历外层验证块。
    outer_training = np.setdiff1d(np.arange(8), outer_validation); inner_validation = outer_training[::2]; inner_training = np.setdiff1d(outer_training, inner_validation); nested_counts.append((len(inner_training), len(inner_validation), len(outer_validation)))  # 只在外层训练集内部再切分。
assert nested_counts == [(2, 2, 4), (2, 2, 4)]  # 综合验证nested CV层级。
