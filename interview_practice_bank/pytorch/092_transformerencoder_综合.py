"""
题目 092：TransformerEncoder_综合

要求：完成“TransformerEncoder”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入Transformer Encoder。
3. 综合原生Encoder、Decoder、padding和causal mask。
4. 验证完整Encoder-Decoder数据流。

完成标准：
- 验证完整Encoder-Decoder数据流。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Transformer Encoder。
full_memory = torch.randn(2, 3, 8); full_targets = torch.randn(2, 4, 8); full_decoder_mask = nn.Transformer.generate_square_subsequent_mask(4); full_decoder_layer = nn.TransformerDecoderLayer(8, 2, 16, batch_first=True, dropout=0.0); full_decoder = nn.TransformerDecoder(full_decoder_layer, 1); encoder_layer_full = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True, norm_first=True, dropout=0.0); encoder_full = nn.TransformerEncoder(encoder_layer_full, 1, enable_nested_tensor=False); source_padding = torch.tensor([[False, False, True], [False, False, False]]); encoded_memory = encoder_full(full_memory, src_key_padding_mask=source_padding); seq2seq_output = full_decoder(full_targets, encoded_memory, tgt_mask=full_decoder_mask)  # 综合原生Encoder、Decoder、padding和causal mask。
assert encoded_memory.shape == full_memory.shape and seq2seq_output.shape == full_targets.shape  # 验证完整Encoder-Decoder数据流。
