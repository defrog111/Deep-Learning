"""
题目 011：线性回归正规方程_易错点

要求：完成“线性回归正规方程”的易错点题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
x = np.arange(1, 6, dtype=float)[:, None]  # 创建单特征样本列。
y = 5 * x[:, 0] + 2  # 按线性规律生成目标。
design = np.c_[np.ones(len(x)), x]  # 添加截距列形成设计矩阵。
weights = np.linalg.pinv(design) @ y  # 使用伪逆求最小二乘解。
predictions = design @ weights  # 计算训练预测。
r_squared = 1 - ((y - predictions) ** 2).sum() / ((y - y.mean()) ** 2).sum()  # 计算R²。
assert np.allclose(weights, [2, 5]) and np.isclose(r_squared, 1)  # 验证恢复真实参数。
print(weights, r_squared)  # 输出回归系数和拟合优度。
