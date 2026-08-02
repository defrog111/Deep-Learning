"""
题目 056：参数初始化_综合

要求：完成“参数初始化”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 综合理解fan_in、fan_out与Kaiming。
4. 验证扇入扇出和随机权重。

完成标准：
- 验证扇入扇出和随机权重。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
fan_tensor = torch.empty(16, 8); fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(fan_tensor); nn.init.kaiming_normal_(fan_tensor, mode='fan_in', nonlinearity='relu')  # 综合理解fan_in、fan_out与Kaiming。
assert (fan_in, fan_out) == (8, 16) and fan_tensor.std() > 0  # 验证扇入扇出和随机权重。
