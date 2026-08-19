"""
题目 006：标准化与归一化_变式

要求：完成“标准化与归一化”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Robust缩放使用中位数和IQR降低离群值影响。
3. 验证中心位于中位数。

完成标准：
- 验证中心位于中位数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
robust_data = np.array([1.0, 2.0, 3.0, 100.0])
robust_scaled = (robust_data - np.median(robust_data)) / (np.percentile(robust_data, 75) - np.percentile(robust_data, 25))  # Robust缩放使用中位数和IQR降低离群值影响。
assert np.median(robust_scaled) == 0  # 验证中心位于中位数。
