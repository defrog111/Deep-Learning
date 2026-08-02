"""
题目 023：自定义nnModule_易错点

要求：完成“自定义nnModule”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入神经网络模块。
3. 普通list中的层不会被Module递归注册。
4. 验证ModuleList注册参数。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
plain_layers = [nn.Linear(2, 2)]; registered_layers = nn.ModuleList([nn.Linear(2, 2)])  # 普通list中的层不会被Module递归注册。
holder = nn.Module(); holder.layers = registered_layers; assert len(list(holder.parameters())) == 2 and len(list(nn.Module().parameters())) == 0  # 验证ModuleList注册参数。
