"""
题目 063：Embedding与padding_idx_易错点

要求：完成“Embedding与padding_idx”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入Embedding模块。
3. padding_idx对应行不累计梯度。
4. 验证padding梯度屏蔽。

完成标准：
- 验证padding梯度屏蔽。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入Embedding模块。
padding_embedding = nn.Embedding(5, 2, padding_idx=0); padding_embedding(torch.tensor([[0, 1]])).sum().backward()  # padding_idx对应行不累计梯度。
assert torch.count_nonzero(padding_embedding.weight.grad[0]) == 0 and torch.count_nonzero(padding_embedding.weight.grad[1]) > 0  # 验证padding梯度屏蔽。
