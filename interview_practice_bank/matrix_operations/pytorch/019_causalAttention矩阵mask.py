"""
题目 019：PyTorch causalAttention矩阵mask

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 用 QK 转置计算 Attention logits。
2. 添加上三角 causal mask。
3. softmax 后验证未来位置概率为零。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import math  # 导入缩放所需平方根函数。
import torch  # 导入 PyTorch。
torch.manual_seed(42)  # 固定随机种子。
query = torch.randn(1, 4, 8)  # 创建 (B,T,D) 查询。
key = torch.randn(1, 4, 8)  # 创建 (B,T,D) 键。
value = torch.randn(1, 4, 6)  # 创建 (B,T,V) 值。
scores = query @ key.transpose(-1, -2) / math.sqrt(8)  # 计算缩放点积 logits。
causal_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)  # True 标记未来位置。
masked_scores = scores.masked_fill(causal_mask, float('-inf'))  # 在 softmax 前把未来 logits 设为负无穷。
weights = masked_scores.softmax(dim=-1)  # 沿 key 轴计算注意力概率。
output = weights @ value  # 加权 V 得到 Attention 输出。
assert torch.count_nonzero(weights.masked_select(causal_mask)) == 0  # 验证未来注意力严格为零。
assert output.shape == (1, 4, 6)  # 验证输出 shape。
print(weights[0], output.shape)  # 输出因果注意力矩阵。
