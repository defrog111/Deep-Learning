"""
题目 090：TransformerEncoder_变式

要求：完成“TransformerEncoder”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入Transformer Encoder。
3. 原生TransformerDecoder包含masked self-attention和cross-attention。
4. 验证Decoder保持目标序列shape。

完成标准：
- 验证Decoder保持目标序列shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Transformer Encoder。
decoder_layer = nn.TransformerDecoderLayer(d_model=8, nhead=2, dim_feedforward=16, batch_first=True, dropout=0.0); decoder = nn.TransformerDecoder(decoder_layer, num_layers=2); target_tokens = torch.randn(2, 4, 8); memory_tokens = torch.randn(2, 3, 8); decoded = decoder(target_tokens, memory_tokens)  # 原生TransformerDecoder包含masked self-attention和cross-attention。
assert decoded.shape == target_tokens.shape  # 验证Decoder保持目标序列shape。
