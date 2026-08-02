"""
题目 021：PyTorch float精度与矩阵乘法

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 比较 float32 与 float64 的病态线性方程。
2. 分别计算解和残差。
3. 理解高精度通常减小数值误差但消耗更多内存。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
a64 = torch.tensor([[1.0, 1.0], [1.0, 1.0001]], dtype=torch.float64)  # 用 float64 创建近奇异矩阵。
b64 = torch.tensor([2.0, 2.0001], dtype=torch.float64)  # 创建精确解应为 [1,1] 的右端。
x64 = torch.linalg.solve(a64, b64)  # 使用双精度求解。
a32, b32 = a64.float(), b64.float()  # 转换为单精度版本。
x32 = torch.linalg.solve(a32, b32)  # 使用单精度求解。
residual64 = torch.linalg.vector_norm(a64 @ x64 - b64)  # 计算双精度残差。
residual32 = torch.linalg.vector_norm(a32 @ x32 - b32)  # 计算单精度残差。
assert torch.allclose(x64, torch.ones(2, dtype=torch.float64), atol=1e-8)  # 验证高精度解。
print(x32, x64, residual32, residual64)  # 比较精度和残差。
