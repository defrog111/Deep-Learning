"""
题目 016：PyTorch 向量矩阵范数与cdist

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 计算 vector_norm 和 matrix_norm。
2. 批量计算两组样本的欧氏距离。
3. 用 normalize 后点积计算余弦相似度。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
from torch.nn import functional as F  # 导入归一化函数。
vector = torch.tensor([3.0, -4.0])  # 创建向量。
matrix = torch.diag(torch.tensor([3.0, 4.0]))  # 创建对角矩阵。
l2 = torch.linalg.vector_norm(vector, ord=2)  # 计算欧氏范数。
frobenius = torch.linalg.matrix_norm(matrix, ord='fro')  # 计算 Frobenius 范数。
left = torch.tensor([[0.0, 0.0], [1.0, 0.0]])  # 创建第一组样本。
right = torch.tensor([[0.0, 1.0], [1.0, 1.0]])  # 创建第二组样本。
distances = torch.cdist(left, right, p=2)  # 计算所有样本对欧氏距离矩阵。
cosine = F.normalize(left + 1e-6, dim=1) @ F.normalize(right + 1e-6, dim=1).t()  # 归一化后点积得到余弦相似度。
assert l2 == frobenius == 5.0 and distances.shape == cosine.shape == (2, 2)  # 核对范数和 shape。
print(l2, frobenius, distances, cosine)  # 输出距离指标。
