"""
题目 059：线性方程组求解_易错点

要求：完成“线性方程组求解”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 定义线性方程系数矩阵A。
2. 定义右侧向量b。
3. 直接求解Ax=b比显式求逆更稳定。
4. 把解代回原方程。
5. 奇异系统不能盲目使用solve，可考虑伪逆。

完成标准：
- 验证数值解。
- 验证伪逆解shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
coefficients = np.array([[3.0, 1.0], [1.0, 2.0]])  # 定义线性方程系数矩阵A。
targets = np.array([12.0, 11.0])  # 定义右侧向量b。
solution = np.linalg.solve(coefficients, targets)  # 直接求解Ax=b比显式求逆更稳定。
reconstructed = coefficients @ solution  # 把解代回原方程。
assert np.allclose(reconstructed, targets)  # 验证数值解。
print(solution, reconstructed)  # 输出解和代回结果。
rank_value = np.linalg.matrix_rank(coefficients); pseudo_solution = np.linalg.pinv(coefficients) @ targets  # 奇异系统不能盲目使用solve，可考虑伪逆。
assert rank_value >= 1 and pseudo_solution.shape[0] == coefficients.shape[1]  # 验证伪逆解shape。
