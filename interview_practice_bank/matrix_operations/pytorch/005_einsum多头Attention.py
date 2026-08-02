"""
题目 005：PyTorch einsum多头Attention

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 创建多头 Q、K、V。
2. 用 einsum 计算缩放点积分数和输出。
3. 检查所有中间 shape。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import math  # 导入平方根函数。
import torch  # 导入 PyTorch。
torch.manual_seed(42)  # 固定随机种子。
query = torch.randn(2, 3, 4, 5)  # 创建 (B,H,Q,D) 查询。
key = torch.randn(2, 3, 6, 5)  # 创建 (B,H,K,D) 键。
value = torch.randn(2, 3, 6, 7)  # 创建 (B,H,K,V) 值。
scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / math.sqrt(5)  # 计算缩放 QK 转置。
weights = scores.softmax(dim=-1)  # 沿 key 轴把分数归一化。
output = torch.einsum('bhqk,bhkv->bhqv', weights, value)  # 用注意力权重加权 V。
assert scores.shape == (2, 3, 4, 6) and output.shape == (2, 3, 4, 7)  # 验证 Attention shape。
assert torch.allclose(weights.sum(-1), torch.ones(2, 3, 4))  # 验证注意力概率和为一。
print(scores.shape, output.shape)  # 输出关键 shape。
