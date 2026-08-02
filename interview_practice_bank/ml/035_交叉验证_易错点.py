"""
题目 035：交叉验证_易错点

要求：完成“交叉验证”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Group CV保证同一主体不跨训练验证。
3. 验证组间隔离。

完成标准：
- 验证组间隔离。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
groups_cv = np.repeat(np.arange(6), 2); validation_group = 2; validation_indices = np.flatnonzero(groups_cv == validation_group); training_indices = np.flatnonzero(groups_cv != validation_group)  # Group CV保证同一主体不跨训练验证。
assert not set(groups_cv[training_indices]) & set(groups_cv[validation_indices])  # 验证组间隔离。
