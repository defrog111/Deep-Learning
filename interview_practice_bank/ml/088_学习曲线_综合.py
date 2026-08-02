"""
题目 088：学习曲线_综合

要求：完成“学习曲线”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合用log-log斜率估计算法随样本量的经验复杂度。
3. 验证二次扩展趋势。

完成标准：
- 验证二次扩展趋势。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
compute_sizes = np.array([10, 20, 40, 80]); fit_times = compute_sizes**2; log_slope = np.polyfit(np.log(compute_sizes), np.log(fit_times), 1)[0]  # 综合用log-log斜率估计算法随样本量的经验复杂度。
assert np.isclose(log_slope, 2)  # 验证二次扩展趋势。
