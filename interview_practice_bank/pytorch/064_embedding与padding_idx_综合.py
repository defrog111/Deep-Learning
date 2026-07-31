"""
题目 064：Embedding与padding_idx_综合

要求：完成“Embedding与padding_idx”的综合题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Embedding模块。
token_ids = torch.tensor([[2, 5, 0, 0], [1, 3, 4, 0]])  # 0作为PAD token。
embedding = nn.Embedding(num_embeddings=10, embedding_dim=12, padding_idx=0)  # 创建词嵌入并固定PAD向量。
vectors = embedding(token_ids)  # 把(B,T)映射为(B,T,D)。
loss = vectors.square().sum()  # 构造用于反向传播的损失。
loss.backward()  # 计算Embedding权重梯度。
assert torch.count_nonzero(embedding.weight.grad[0]) == 0  # padding_idx对应行不更新。
print(vectors.shape, embedding.weight.grad[0])  # 输出shape和PAD梯度。
