"""
题目 046：类别不平衡与阈值_变式

要求：完成“类别不平衡与阈值”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 只在训练集内随机过采样少数类。
3. 验证有放回过采样。

完成标准：
- 验证有放回过采样。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
majority = np.arange(20); minority = np.arange(20, 24); rng_balance = np.random.default_rng(42); oversampled_minority = rng_balance.choice(minority, size=len(majority), replace=True); balanced_indices = np.r_[majority, oversampled_minority]  # 只在训练集内随机过采样少数类。
assert len(balanced_indices) == 40 and len(np.unique(oversampled_minority)) <= len(minority)  # 验证有放回过采样。
