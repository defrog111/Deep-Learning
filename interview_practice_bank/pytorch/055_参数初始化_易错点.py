"""
题目 055：参数初始化_易错点

要求：完成“参数初始化”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 导入模型模块。
3. 正交初始化使方阵W乘WT接近单位阵。
4. 验证正交性。

完成标准：
- 验证正交性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
orthogonal_layer = nn.Linear(4, 4, bias=False); nn.init.orthogonal_(orthogonal_layer.weight)  # 正交初始化使方阵W乘WT接近单位阵。
assert torch.allclose(orthogonal_layer.weight @ orthogonal_layer.weight.T, torch.eye(4), atol=1e-5)  # 验证正交性。
