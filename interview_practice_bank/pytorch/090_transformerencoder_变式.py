"""
题目 090：TransformerEncoder_变式

要求：完成“TransformerEncoder”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 配置单层Encoder。
3. 堆叠随变式变化的层数。
4. 创建(B,T,D)输入。
5. 定义PAD mask。
6. 编码时屏蔽PAD key。
7. masked mean pooling。

完成标准：
- 验证序列与池化shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Transformer Encoder。
torch.manual_seed(0)  # 固定随机种子。
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, batch_first=True)  # 配置单层Encoder。
encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)  # 堆叠随变式变化的层数。
tokens = torch.randn(2, 6, 16)  # 创建(B,T,D)输入。
padding_mask = torch.tensor([[False, False, False, False, True, True], [False, False, False, False, False, True]])  # 定义PAD mask。
encoded = encoder(tokens, src_key_padding_mask=padding_mask)  # 编码时屏蔽PAD key。
pooled = (encoded * (~padding_mask).unsqueeze(-1)).sum(1) / (~padding_mask).sum(1, keepdim=True)  # masked mean pooling。
assert encoded.shape == tokens.shape and pooled.shape == (2, 16)  # 验证序列与池化shape。
print(encoded.shape, pooled.shape)  # 输出Encoder结果。
