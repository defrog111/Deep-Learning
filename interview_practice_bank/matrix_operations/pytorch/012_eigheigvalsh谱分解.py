"""
题目 012：PyTorch eigheigvalsh谱分解

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 对实对称矩阵调用 eigh。
2. 验证特征方程和正交性。
3. 只需要特征值时调用 eigvalsh。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]])  # 创建实对称矩阵。
values, vectors = torch.linalg.eigh(matrix)  # 取得升序实特征值和正交特征向量。
values_only = torch.linalg.eigvalsh(matrix)  # 不计算特征向量可节省工作。
reconstructed = vectors @ torch.diag(values) @ vectors.t()  # 用 QΛQ 转置重建矩阵。
assert torch.allclose(matrix @ vectors, vectors * values)  # 验证 Av=lambda*v。
assert torch.allclose(values, values_only) and torch.allclose(reconstructed, matrix)  # 核对特征值和重建。
print(values, vectors)  # 输出谱分解结果。
