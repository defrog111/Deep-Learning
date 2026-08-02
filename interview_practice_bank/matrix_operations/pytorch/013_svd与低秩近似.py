"""
题目 013：PyTorch svd与低秩近似

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 调用 torch.linalg.svd。
2. 完整重建矩阵。
3. 用最大奇异值构造 rank-1 近似。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
matrix = torch.tensor([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])  # 创建非方阵。
u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)  # 计算经济型 SVD。
reconstructed = (u * singular) @ vh  # 使用广播完成 UΣVh 重建。
rank_one = (u[:, :1] * singular[:1]) @ vh[:1]  # 只保留最大奇异值。
full_error = torch.linalg.matrix_norm(matrix - reconstructed)  # 计算完整重建 Frobenius 误差。
rank_one_error = torch.linalg.matrix_norm(matrix - rank_one)  # 计算低秩近似误差。
assert full_error < 1e-5 and rank_one_error > full_error  # 验证重建与压缩。
print(singular, rank_one, rank_one_error)  # 输出奇异值和近似。
