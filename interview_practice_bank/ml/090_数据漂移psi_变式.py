"""
题目 090：数据漂移PSI_变式

要求：完成“数据漂移PSI”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Total Variation衡量离散分布整体差异。
3. 验证漂移距离。

完成标准：
- 验证漂移距离。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
reference_categories = np.array([0.5, 0.3, 0.2]); current_categories = np.array([0.4, 0.4, 0.2]); total_variation = 0.5 * np.abs(reference_categories - current_categories).sum()  # Total Variation衡量离散分布整体差异。
assert np.isclose(total_variation, 0.1)  # 验证漂移距离。
