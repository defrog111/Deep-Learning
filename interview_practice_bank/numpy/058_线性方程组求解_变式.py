"""
题目 058：线性方程组求解_变式

要求：完成“线性方程组求解”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 定义线性方程系数矩阵A。
2. 定义右侧向量b。
3. 非方阵或含噪问题使用最小二乘。
4. 检查秩和奇异值。

完成标准：
- 检查秩和奇异值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
coefficients = np.array([[3.0, 1.0], [1.0, 2.0]])  # 定义线性方程系数矩阵A。
targets = np.array([11.0, 10.0])  # 定义右侧向量b。
least_squares, residuals, rank, singular = np.linalg.lstsq(coefficients, targets, rcond=None)  # 非方阵或含噪问题使用最小二乘。
assert rank <= min(coefficients.shape) and singular.ndim == 1  # 检查秩和奇异值。
