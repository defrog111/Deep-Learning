"""
题目 088：多项式拟合_综合

要求：完成“多项式拟合”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 用独立数据和Vandermonde设计矩阵实现拟合。
3. 验证底层最小二乘系数和预测。

完成标准：
- 验证底层最小二乘系数和预测。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
sample_x = np.linspace(-1, 1, 7); sample_y = 3 * sample_x**2 - 2 * sample_x + 4; vandermonde = np.vander(sample_x, N=3); manual_coefficients = np.linalg.lstsq(vandermonde, sample_y, rcond=None)[0]  # 用独立数据和Vandermonde设计矩阵实现拟合。
assert np.allclose(manual_coefficients, [3, -2, 4]) and np.allclose(vandermonde @ manual_coefficients, sample_y)  # 验证底层最小二乘系数和预测。
