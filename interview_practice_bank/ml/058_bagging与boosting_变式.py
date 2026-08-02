"""
题目 058：Bagging与Boosting_变式

要求：完成“Bagging与Boosting”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Bagging有放回抽样，未抽中样本形成OOB集。
3. 验证bootstrap与OOB。

完成标准：
- 验证bootstrap与OOB。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng_bootstrap = np.random.default_rng(1); bootstrap_indices = rng_bootstrap.choice(10, size=10, replace=True); oob_indices = np.setdiff1d(np.arange(10), np.unique(bootstrap_indices))  # Bagging有放回抽样，未抽中样本形成OOB集。
assert len(np.unique(bootstrap_indices)) <= 10 and not set(bootstrap_indices) & set(oob_indices)  # 验证bootstrap与OOB。
