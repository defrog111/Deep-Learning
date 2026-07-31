"""
题目 087：多项式拟合_易错点

要求：完成“多项式拟合”的易错点题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
x = np.linspace(-2, 2, 9)  # 创建自变量采样点。
y = 2 * x**2 - 3 * x + 1  # 根据二次函数生成无噪声目标。
coefficients = np.polyfit(x, y, deg=2)  # 最小二乘拟合二次多项式。
predictions = np.polyval(coefficients, x)  # 用拟合系数计算预测。
assert np.allclose(predictions, y)  # 无噪声且阶数正确时应精确拟合。
print(coefficients, np.mean((predictions - y) ** 2))  # 输出系数和MSE。
