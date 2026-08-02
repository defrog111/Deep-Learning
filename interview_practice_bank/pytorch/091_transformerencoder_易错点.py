"""
题目 091：TransformerEncoder_易错点

要求：完成“TransformerEncoder”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入Transformer Encoder。
3. causal mask和padding mask都用bool，避免类型不匹配警告。
4. 区分两个mask的shape和用途。

完成标准：
- 区分两个mask的shape和用途。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Transformer Encoder。
masked_decoder_layer = nn.TransformerDecoderLayer(8, 2, 16, batch_first=True, dropout=0.0); masked_decoder = nn.TransformerDecoder(masked_decoder_layer, 1); masked_targets = torch.randn(2, 4, 8); masked_memory = torch.randn(2, 3, 8); decoder_mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1); decoder_padding = torch.tensor([[False, False, True, True], [False, False, False, True]]); masked_decoded = masked_decoder(masked_targets, masked_memory, tgt_mask=decoder_mask, tgt_key_padding_mask=decoder_padding)  # causal mask和padding mask都用bool，避免类型不匹配警告。
assert decoder_mask.shape == (4, 4) and masked_decoded.shape == masked_targets.shape  # 区分两个mask的shape和用途。
