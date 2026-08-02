"""
题目 006：PyTorch outerger与addmm

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算向量外积。
2. 用 ger 验证 outer 的旧别名行为。
3. 用 addmm 融合偏置和矩阵乘法。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
a = torch.tensor([1.0, 2.0, 3.0])  # 创建列方向向量。
b = torch.tensor([4.0, 5.0])  # 创建行方向向量。
outer = torch.outer(a, b)  # 计算每对元素乘积得到 (3,2)。
via_ger = torch.ger(a, b)  # ger 是二维外积的旧接口。
left = torch.ones(3, 4)  # 创建 addmm 的左矩阵。
right = torch.ones(4, 2)  # 创建 addmm 的右矩阵。
fused = torch.addmm(outer, left, right, beta=0.5, alpha=2.0)  # 融合计算 0.5*outer+2*(left@right)。
assert torch.equal(outer, via_ger) and torch.allclose(fused, 0.5 * outer + 2 * (left @ right))  # 核对结果。
print(outer, fused)  # 输出外积与融合结果。
