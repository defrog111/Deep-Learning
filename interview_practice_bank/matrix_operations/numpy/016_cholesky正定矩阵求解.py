"""
题目 016：NumPy cholesky正定矩阵求解

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 验证矩阵对称正定。
2. 计算 Cholesky 分解。
3. 用两次三角方程求解 Ax=b。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
a = np.array([[4.0, 2.0], [2.0, 3.0]])  # 创建对称正定矩阵。
b = np.array([6.0, 7.0])  # 创建右端向量。
lower = np.linalg.cholesky(a)  # 得到 A=L@L.T 的下三角因子。
intermediate = np.linalg.solve(lower, b)  # 先求 Ly=b。
x = np.linalg.solve(lower.T, intermediate)  # 再求 L.T x=y。
assert np.all(np.linalg.eigvalsh(a) > 0) and np.allclose(lower @ lower.T, a)  # 验证正定性和分解。
assert np.allclose(a @ x, b)  # 代回原方程验证。
print(lower, x)  # 输出 Cholesky 因子与解。
