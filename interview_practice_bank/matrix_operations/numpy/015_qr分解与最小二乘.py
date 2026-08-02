"""
题目 015：NumPy qr分解与最小二乘

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算 reduced QR 分解。
2. 检查 Q 列正交并重建 A。
3. 用 QR 解满列秩最小二乘。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
a = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])  # 创建高矩阵设计矩阵。
b = np.array([1.0, 2.0, 2.0])  # 创建回归目标。
q, r = np.linalg.qr(a, mode='reduced')  # 分解 A=QR，其中 Q 的列正交。
x_via_qr = np.linalg.solve(r, q.T @ b)  # 求解 Rx=Q 转置 b。
x_via_lstsq = np.linalg.lstsq(a, b, rcond=None)[0]  # 使用标准最小二乘作为对照。
assert np.allclose(q.T @ q, np.eye(2)) and np.allclose(q @ r, a)  # 验证正交性和重建。
assert np.allclose(x_via_qr, x_via_lstsq)  # 核对 QR 解。
print(q, r, x_via_qr)  # 输出 QR 因子与解。
