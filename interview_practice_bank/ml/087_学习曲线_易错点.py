"""
题目 087：学习曲线_易错点

要求：完成“学习曲线”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 两条曲线接近但都差表示高偏差。
3. 验证不能只看gap。

完成标准：
- 验证不能只看gap。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
underfit_train = np.array([0.55, 0.57, 0.58]); underfit_valid = np.array([0.53, 0.55, 0.56]); small_gap = np.abs(underfit_train[-1] - underfit_valid[-1]); low_both = max(underfit_train[-1], underfit_valid[-1]) < 0.7  # 两条曲线接近但都差表示高偏差。
assert small_gap < 0.05 and low_both  # 验证不能只看gap。
