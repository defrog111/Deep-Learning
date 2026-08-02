"""
题目 011：NumPy 伪逆与秩亏矩阵

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 构造线性相关列导致的秩亏矩阵。
2. 说明 solve 为什么不适用。
3. 使用 pinv 求最小范数解。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
a = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # 第二列是第一列两倍，因此矩阵秩亏。
b = np.array([1.0, 2.0, 3.0])  # 创建与列空间一致的目标。
rank = np.linalg.matrix_rank(a)  # 计算秩并识别线性相关列。
x = np.linalg.pinv(a) @ b  # Moore-Penrose 伪逆给出最小范数最小二乘解。
assert rank == 1 and np.allclose(a @ x, b)  # 验证秩和重建结果。
assert np.allclose(x, np.array([0.2, 0.4]))  # 核对最小范数解。
print(rank, x)  # 输出秩和伪逆解。
