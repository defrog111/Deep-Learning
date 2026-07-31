"""
题目 062：Embedding与padding_idx_变式

要求：完成“Embedding与padding_idx”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 0作为PAD token。
2. 创建词嵌入并固定PAD向量。
3. 把(B,T)映射为(B,T,D)。
4. 构造用于反向传播的损失。
5. 计算Embedding权重梯度。
6. padding_idx对应行不更新。
7. EmbeddingBag无需显式padding即可聚合集合式token。

完成标准：
- padding_idx对应行不更新。
- 验证两个bag的输出shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Embedding模块。
token_ids = torch.tensor([[2, 5, 0, 0], [1, 3, 4, 0]])  # 0作为PAD token。
embedding = nn.Embedding(num_embeddings=10, embedding_dim=8, padding_idx=0)  # 创建词嵌入并固定PAD向量。
vectors = embedding(token_ids)  # 把(B,T)映射为(B,T,D)。
loss = vectors.square().sum()  # 构造用于反向传播的损失。
loss.backward()  # 计算Embedding权重梯度。
assert torch.count_nonzero(embedding.weight.grad[0]) == 0  # padding_idx对应行不更新。
print(vectors.shape, embedding.weight.grad[0])  # 输出shape和PAD梯度。
embedding_bag = nn.EmbeddingBag(6, 3, mode='mean'); flat_tokens = torch.tensor([1, 2, 3, 4]); offsets = torch.tensor([0, 2]); bag_output = embedding_bag(flat_tokens, offsets)  # EmbeddingBag无需显式padding即可聚合集合式token。
assert bag_output.shape == (2, 3)  # 验证两个bag的输出shape。
