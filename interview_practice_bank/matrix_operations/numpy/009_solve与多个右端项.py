"""
题目 009：NumPy solve与多个右端项

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 求解 Ax=b。
2. 一次求解多个右端向量。
3. 代回原方程验证，并说明不要先求逆。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
a = np.array([[3.0, 1.0], [1.0, 2.0]])  # 创建可逆系数矩阵 A。
b = np.array([9.0, 8.0])  # 创建单个右端向量 b。
x = np.linalg.solve(a, b)  # 直接求解比 inv(A)@b 更快且更稳定。
multiple_b = np.column_stack((b, 2 * b))  # 把两个右端向量组成 shape=(2,2)。
multiple_x = np.linalg.solve(a, multiple_b)  # 一次求出两组解。
assert np.allclose(a @ x, b) and np.allclose(a @ multiple_x, multiple_b)  # 把解代回验证。
print(x, multiple_x)  # 输出单组和多组解。
