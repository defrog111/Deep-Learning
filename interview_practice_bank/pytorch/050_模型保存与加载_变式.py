"""
题目 050：模型保存与加载_变式

要求：完成“模型保存与加载”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入路径工具。
2. 导入临时目录工具。
3. 导入 PyTorch。
4. 导入模型模块。
5. 完整checkpoint保存轮数、模型和优化器。
6. 验证恢复训练必需字段。

完成标准：
- 验证恢复训练必需字段。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import tempfile  # 导入临时目录工具。
import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
checkpoint_model = nn.Linear(2, 1); checkpoint_optimizer = torch.optim.SGD(checkpoint_model.parameters(), lr=0.1, momentum=0.9); checkpoint = {'epoch': 3, 'model': checkpoint_model.state_dict(), 'optimizer': checkpoint_optimizer.state_dict()}  # 完整checkpoint保存轮数、模型和优化器。
assert checkpoint['epoch'] == 3 and set(checkpoint) == {'epoch', 'model', 'optimizer'}  # 验证恢复训练必需字段。
