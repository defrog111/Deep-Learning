"""
题目 091：TransformerEncoder_易错点

要求：完成“TransformerEncoder”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Transformer Encoder。
torch.manual_seed(0)  # 固定随机种子。
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, batch_first=True)  # 配置单层Encoder。
encoder = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)  # 堆叠随变式变化的层数。
tokens = torch.randn(2, 6, 16)  # 创建(B,T,D)输入。
padding_mask = torch.tensor([[False, False, False, False, True, True], [False, False, False, False, False, True]])  # 定义PAD mask。
encoded = encoder(tokens, src_key_padding_mask=padding_mask)  # 编码时屏蔽PAD key。
pooled = (encoded * (~padding_mask).unsqueeze(-1)).sum(1) / (~padding_mask).sum(1, keepdim=True)  # masked mean pooling。
assert encoded.shape == tokens.shape and pooled.shape == (2, 16)  # 验证序列与池化shape。
print(encoded.shape, pooled.shape)  # 输出Encoder结果。
