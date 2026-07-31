"""
题目 057：线性方程组求解_基础

要求：完成“线性方程组求解”的基础题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
coefficients = np.array([[3.0, 1.0], [1.0, 2.0]])  # 定义线性方程系数矩阵A。
targets = np.array([10.0, 9.0])  # 定义右侧向量b。
solution = np.linalg.solve(coefficients, targets)  # 直接求解Ax=b比显式求逆更稳定。
reconstructed = coefficients @ solution  # 把解代回原方程。
assert np.allclose(reconstructed, targets)  # 验证数值解。
print(solution, reconstructed)  # 输出解和代回结果。
