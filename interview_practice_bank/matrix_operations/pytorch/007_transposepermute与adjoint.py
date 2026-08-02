"""
题目 007：PyTorch transposepermute与adjoint

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 区分 t、transpose 和 permute。
2. 只交换批量矩阵的最后两轴。
3. 对复数矩阵做共轭转置。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
batch = torch.arange(24).reshape(2, 3, 4)  # 创建三维批量矩阵。
last_two = batch.transpose(-1, -2)  # 只交换最后两个矩阵轴得到 (2,4,3)。
reordered = batch.permute(2, 0, 1)  # 任意重排全部轴得到 (4,2,3)。
complex_matrix = torch.tensor([[1 + 2j, 3 - 1j]])  # 创建复数矩阵。
adjoint = complex_matrix.mH  # mH 对最后两轴执行共轭转置。
assert last_two.shape == (2, 4, 3) and reordered.shape == (4, 2, 3)  # 验证轴顺序。
assert torch.equal(adjoint, complex_matrix.conj().transpose(-1, -2))  # 核对共轭转置。
print(last_two.shape, reordered.shape, adjoint)  # 输出换轴结果。
