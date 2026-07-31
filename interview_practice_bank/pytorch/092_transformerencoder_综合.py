"""
题目 092：TransformerEncoder_综合

要求：完成“TransformerEncoder”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 配置单层Encoder。
3. 堆叠随变式变化的层数。
4. 创建(B,T,D)输入。
5. 定义PAD mask。
6. 编码时屏蔽PAD key。
7. masked mean pooling。
8. 综合原生Encoder、Decoder、padding和causal mask。

完成标准：
- 验证序列与池化shape。
- 验证完整Encoder-Decoder数据流。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Transformer Encoder。
torch.manual_seed(0)  # 固定随机种子。
layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=64, batch_first=True)  # 配置单层Encoder。
encoder = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)  # 堆叠随变式变化的层数。
tokens = torch.randn(2, 6, 16)  # 创建(B,T,D)输入。
padding_mask = torch.tensor([[False, False, False, False, True, True], [False, False, False, False, False, True]])  # 定义PAD mask。
encoded = encoder(tokens, src_key_padding_mask=padding_mask)  # 编码时屏蔽PAD key。
pooled = (encoded * (~padding_mask).unsqueeze(-1)).sum(1) / (~padding_mask).sum(1, keepdim=True)  # masked mean pooling。
assert encoded.shape == tokens.shape and pooled.shape == (2, 16)  # 验证序列与池化shape。
print(encoded.shape, pooled.shape)  # 输出Encoder结果。
full_memory = torch.randn(2, 3, 8); full_targets = torch.randn(2, 4, 8); full_decoder_mask = nn.Transformer.generate_square_subsequent_mask(4); full_decoder_layer = nn.TransformerDecoderLayer(8, 2, 16, batch_first=True, dropout=0.0); full_decoder = nn.TransformerDecoder(full_decoder_layer, 1); encoder_layer_full = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True, norm_first=True, dropout=0.0); encoder_full = nn.TransformerEncoder(encoder_layer_full, 1, enable_nested_tensor=False); source_padding = torch.tensor([[False, False, True], [False, False, False]]); encoded_memory = encoder_full(full_memory, src_key_padding_mask=source_padding); seq2seq_output = full_decoder(full_targets, encoded_memory, tgt_mask=full_decoder_mask)  # 综合原生Encoder、Decoder、padding和causal mask。
assert encoded_memory.shape == full_memory.shape and seq2seq_output.shape == full_targets.shape  # 验证完整Encoder-Decoder数据流。
