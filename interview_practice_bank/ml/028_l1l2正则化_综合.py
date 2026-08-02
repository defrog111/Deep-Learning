"""
题目 028：L1L2正则化_综合

要求：完成“L1L2正则化”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 正交设计下Lasso等价于软阈值。
3. 综合验证L1产生稀疏系数。

完成标准：
- 综合验证L1产生稀疏系数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
orthogonal_design = np.eye(3); sparse_target = np.array([3.0, 0.1, -2.0]); threshold = 0.5; lasso_closed_form = np.sign(sparse_target) * np.maximum(np.abs(sparse_target) - threshold, 0)  # 正交设计下Lasso等价于软阈值。
assert lasso_closed_form.tolist() == [2.5, 0.0, -1.5]  # 综合验证L1产生稀疏系数。
