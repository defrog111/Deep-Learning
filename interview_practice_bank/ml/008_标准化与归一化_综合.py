"""
题目 008：标准化与归一化_综合

要求：完成“标准化与归一化”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合先log1p减小偏态再标准化。
3. 验证组合变换。

完成标准：
- 验证组合变换。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
skewed = np.array([0.0, 1.0, 9.0, 99.0]); log_transformed = np.log1p(skewed); standardized_log = (log_transformed - log_transformed.mean()) / log_transformed.std()  # 综合先log1p减小偏态再标准化。
assert np.isclose(standardized_log.mean(), 0) and np.isclose(standardized_log.std(), 1)  # 验证组合变换。
