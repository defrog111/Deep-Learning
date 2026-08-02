"""
题目 012：线性回归正规方程_综合

要求：完成“线性回归正规方程”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合拟合并检查残差。
3. 含截距OLS残差均值接近零。

完成标准：
- 含截距OLS残差均值接近零。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
diagnostic_x = np.arange(1, 8.0); diagnostic_y = 2 * diagnostic_x + np.array([-1, 0, 1, 0, -1, 0, 1]); diagnostic_coef = np.polyfit(diagnostic_x, diagnostic_y, 1); diagnostic_residuals = diagnostic_y - np.polyval(diagnostic_coef, diagnostic_x)  # 综合拟合并检查残差。
assert abs(diagnostic_residuals.mean()) < 1e-12 and diagnostic_residuals.shape == diagnostic_y.shape  # 含截距OLS残差均值接近零。
