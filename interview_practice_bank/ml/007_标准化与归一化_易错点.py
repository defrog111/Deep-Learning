"""
题目 007：标准化与归一化_易错点

要求：完成“标准化与归一化”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 常数列标准差为零，必须防止除零。
3. 验证安全处理。

完成标准：
- 验证安全处理。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
constant_feature = np.ones(4); safe_denominator = np.where(constant_feature.std() == 0, 1.0, constant_feature.std()); safe_standardized = (constant_feature - constant_feature.mean()) / safe_denominator  # 常数列标准差为零，必须防止除零。
assert np.isfinite(safe_standardized).all() and np.allclose(safe_standardized, 0)  # 验证安全处理。
