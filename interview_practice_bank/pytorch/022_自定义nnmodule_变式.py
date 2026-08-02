"""
题目 022：自定义nnModule_变式

要求：完成“自定义nnModule”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入神经网络模块。
3. 演示Parameter和buffer的注册差异。
    def __init__(self):  # 初始化模块。
        super().__init__()  # 注册Module内部结构。
        self.weight = nn.Parameter(torch.ones(2))  # Parameter参与优化。
        self.register_buffer('running', torch.zeros(2))  # buffer随模型保存和迁移但不求梯度。
4. 验证注册结果。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
class RegisteredState(nn.Module):  # 演示Parameter和buffer的注册差异。
    def __init__(self):  # 初始化模块。
        super().__init__()  # 注册Module内部结构。
        self.weight = nn.Parameter(torch.ones(2))  # Parameter参与优化。
        self.register_buffer('running', torch.zeros(2))  # buffer随模型保存和迁移但不求梯度。
registered = RegisteredState(); assert list(dict(registered.named_parameters())) == ['weight'] and list(dict(registered.named_buffers())) == ['running']  # 验证注册结果。
