"""
题目 012：线性回归正规方程_综合

要求：完成“线性回归正规方程”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建单特征样本列。
2. 按线性规律生成目标。
3. 添加截距列形成设计矩阵。
4. 使用伪逆求最小二乘解。
5. 计算训练预测。
6. 计算R²。
7. 综合拟合并检查残差。
8. 含截距OLS残差均值接近零。

完成标准：
- 验证恢复真实参数。
- 含截距OLS残差均值接近零。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
x = np.arange(1, 6, dtype=float)[:, None]  # 创建单特征样本列。
y = 6 * x[:, 0] + 2  # 按线性规律生成目标。
design = np.c_[np.ones(len(x)), x]  # 添加截距列形成设计矩阵。
weights = np.linalg.pinv(design) @ y  # 使用伪逆求最小二乘解。
predictions = design @ weights  # 计算训练预测。
r_squared = 1 - ((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # 计算R²。
assert np.allclose(weights, [2, 6]) and np.isclose(r_squared, 1)  # 验证恢复真实参数。
print(weights, r_squared)  # 输出回归系数和拟合优度。
diagnostic_x = np.arange(1, 8.0); diagnostic_y = 2 * diagnostic_x + np.array([-1, 0, 1, 0, -1, 0, 1]); diagnostic_coef = np.polyfit(diagnostic_x, diagnostic_y, 1); diagnostic_residuals = diagnostic_y - np.polyval(diagnostic_coef, diagnostic_x)  # 综合拟合并检查残差。
assert abs(diagnostic_residuals.mean()) < 1e-12 and diagnostic_residuals.shape == diagnostic_y.shape  # 含截距OLS残差均值接近零。
