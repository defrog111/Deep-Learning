"""
题目 008：PyTorch linalgsolve与批量方程

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 求解单个线性方程组。
2. 扩展到 batch 线性方程。
3. 代回验证并说明 solve 优于显式求逆。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
a = torch.tensor([[3.0, 1.0], [1.0, 2.0]])  # 创建可逆系数矩阵。
b = torch.tensor([9.0, 8.0])  # 创建右端向量。
x = torch.linalg.solve(a, b)  # 直接求解 Ax=b。
batch_a = torch.stack((a, 2 * a))  # 创建两个方程组的批量系数矩阵。
batch_b = torch.stack((b, 2 * b))  # 创建对应批量右端向量。
batch_x = torch.linalg.solve(batch_a, batch_b)  # 一次求解 batch 方程。
assert torch.allclose(a @ x, b)  # 验证单组解。
assert torch.allclose(batch_a @ batch_x.unsqueeze(-1), batch_b.unsqueeze(-1))  # 验证批量解。
print(x, batch_x)  # 输出两类解。
