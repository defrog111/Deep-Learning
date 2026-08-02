"""
题目 003：PyTorch bmm与baddbmm

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 用 bmm 完成严格三维批量乘法。
2. 用 baddbmm 同时加入偏置。
3. 说明 bmm 本身不广播 batch。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
left = torch.arange(24.0).reshape(2, 3, 4)  # 创建 (B,M,K) 左 Tensor。
right = torch.ones(2, 4, 5)  # 创建同 batch 的 (B,K,N) 右 Tensor。
product = torch.bmm(left, right)  # 对每个 batch 严格执行矩阵乘法。
bias = torch.full((2, 3, 5), 2.0)  # 创建与输出同 shape 的偏置。
combined = torch.baddbmm(bias, left, right, beta=0.5, alpha=2.0)  # 计算 beta*bias+alpha*(left@right)。
expected = 0.5 * bias + 2.0 * product  # 直接写出等价公式。
assert product.shape == (2, 3, 5) and torch.allclose(combined, expected)  # 核对 shape 和数值。
print(product.shape, combined[0])  # 输出批量结果。
