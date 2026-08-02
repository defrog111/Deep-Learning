"""
题目 062：Embedding与padding_idx_变式

要求：完成“Embedding与padding_idx”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入Embedding模块。
3. EmbeddingBag无需显式padding即可聚合集合式token。
4. 验证两个bag的输出shape。

完成标准：
- 验证两个bag的输出shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Embedding模块。
embedding_bag = nn.EmbeddingBag(6, 3, mode='mean'); flat_tokens = torch.tensor([1, 2, 3, 4]); offsets = torch.tensor([0, 2]); bag_output = embedding_bag(flat_tokens, offsets)  # EmbeddingBag无需显式padding即可聚合集合式token。
assert bag_output.shape == (2, 3)  # 验证两个bag的输出shape。
