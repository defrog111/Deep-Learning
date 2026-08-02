"""
题目 020：PyTorch 稀疏矩阵乘法

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 用 COO 索引和值创建稀疏矩阵。
2. 执行 sparse.mm。
3. 转成 dense 后核对并查看稀疏梯度流。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import torch  # 导入 PyTorch。
indices = torch.tensor([[0, 1, 1], [2, 0, 2]])  # 每列描述一个非零元素的行列坐标。
values = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)  # 创建可训练非零值。
torch.sparse.check_sparse_tensor_invariants.enable()  # 显式开启 COO 索引和值约束检查。
sparse = torch.sparse_coo_tensor(indices, values, size=(2, 3)).coalesce()  # 构建并合并 COO 稀疏矩阵。
dense_right = torch.arange(6.0).reshape(3, 2)  # 创建稠密右矩阵。
output = torch.sparse.mm(sparse, dense_right)  # 执行稀疏乘稠密矩阵。
expected = sparse.to_dense() @ dense_right  # 转稠密计算对照结果。
output.sum().backward()  # 验证非零 values 可以参与 Autograd。
assert torch.allclose(output, expected) and values.grad is not None  # 核对结果和梯度。
print(sparse, output, values.grad)  # 输出稀疏结构、乘积和梯度。
