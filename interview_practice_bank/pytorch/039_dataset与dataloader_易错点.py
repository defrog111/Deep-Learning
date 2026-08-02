"""
题目 039：Dataset与DataLoader_易错点

要求：完成“Dataset与DataLoader”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入数据集和批加载器。
3. 导入神经网络和序列工具。
4. 自定义collate处理变长序列。
5. 验证动态padding。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch.utils.data import DataLoader, TensorDataset  # 导入数据集和批加载器。
from torch import nn  # 导入神经网络和序列工具。
variable_sequences = [torch.arange(2), torch.arange(4), torch.arange(3)]; variable_loader = DataLoader(variable_sequences, batch_size=3, collate_fn=lambda batch: nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=-1))  # 自定义collate处理变长序列。
padded_batch = next(iter(variable_loader)); assert padded_batch.shape == (3, 4) and padded_batch[0, -1] == -1  # 验证动态padding。
