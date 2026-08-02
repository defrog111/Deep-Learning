"""
题目 015：PyTorch cholesky与choleskysolve

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 构造对称正定矩阵。
2. 计算 Cholesky 因子。
3. 使用 cholesky_solve 求多右端项。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
a = torch.tensor([[4.0, 2.0], [2.0, 3.0]])  # 创建对称正定矩阵。
b = torch.tensor([[6.0], [7.0]])  # cholesky_solve 接收二维右端项。
lower = torch.linalg.cholesky(a)  # 得到 A=L@L 转置。
x = torch.cholesky_solve(b, lower, upper=False)  # 利用下三角 Cholesky 因子求解。
assert torch.all(torch.linalg.eigvalsh(a) > 0)  # 验证正定性。
assert torch.allclose(lower @ lower.t(), a) and torch.allclose(a @ x, b)  # 验证分解和解。
print(lower, x)  # 输出因子和解。
