"""例 01：token embedding、缩放和 shape。"""

import math

import torch
from torch import nn


torch.manual_seed(0)
token_ids = torch.tensor([[1, 4, 2, 0], [3, 2, 5, 1]])  # (B=2, T=4)
embedding = nn.Embedding(num_embeddings=10, embedding_dim=8, padding_idx=0)
token_vectors = embedding(token_ids) * math.sqrt(8)  # (2, 4, 8)

print("token ids:", token_ids.shape)
print("token vectors:", token_vectors.shape)
print("padding token vector:", token_vectors[0, 3])
