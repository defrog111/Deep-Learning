"""
题目 002：PyTorch matmul维度与广播

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 比较 1D、2D 和 N-D matmul 规则。
2. 让共享权重沿 batch 维广播。
3. 验证输出 shape。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
batch = torch.arange(24.0).reshape(2, 3, 4)  # 创建 (B=2,M=3,K=4) 批量矩阵。
weight = torch.ones(4, 5)  # 创建共享 (K=4,N=5) 权重。
output = torch.matmul(batch, weight)  # N-D matmul 广播 batch 维并得到 (2,3,5)。
vector_output = torch.matmul(batch, torch.ones(4))  # 批量矩阵乘向量得到 (2,3)。
assert output.shape == (2, 3, 5) and vector_output.shape == (2, 3)  # 验证不同维度规则。
print(output.shape, vector_output)  # 输出批量矩阵结果。
