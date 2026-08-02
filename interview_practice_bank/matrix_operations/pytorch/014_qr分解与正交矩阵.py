"""
题目 014：PyTorch qr分解与正交矩阵

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算 reduced QR。
2. 验证 Q 转置 Q 等于单位矩阵。
3. 使用 QR 解最小二乘并与 lstsq 核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
a = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])  # 创建高矩阵。
b = torch.tensor([1.0, 2.0, 2.0])  # 创建回归目标。
q, r = torch.linalg.qr(a, mode='reduced')  # 分解 A=QR。
x_qr = torch.linalg.solve(r, q.t() @ b)  # 通过 Rx=Q 转置 b 求最小二乘。
x_lstsq = torch.linalg.lstsq(a, b).solution  # 用官方最小二乘核对。
assert torch.allclose(q.t() @ q, torch.eye(2), atol=1e-6) and torch.allclose(q @ r, a)  # 验证 Q 正交和重建。
assert torch.allclose(x_qr, x_lstsq, atol=1e-5)  # 核对两种解。
print(q, r, x_qr)  # 输出 QR 分解与解。
