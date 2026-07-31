"""
题目 055：参数初始化_易错点

要求：完成“参数初始化”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. 固定随机种子。
2. 创建线性层。
3. 为ReLU网络使用Kaiming初始化变式3。
4. 偏置初始化为零。
5. 取得扇入扇出用于解释方差。
6. 正交初始化使方阵W乘WT接近单位阵。

完成标准：
- 验证偏置为零。
- 验证正交性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
torch.manual_seed(0)  # 固定随机种子。
layer = nn.Linear(64, 32)  # 创建线性层。
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')  # 为ReLU网络使用Kaiming初始化变式3。
nn.init.zeros_(layer.bias)  # 偏置初始化为零。
fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight)  # 取得扇入扇出用于解释方差。
assert torch.count_nonzero(layer.bias) == 0  # 验证偏置为零。
print(layer.weight.mean().item(), layer.weight.std().item(), fan_in, fan_out)  # 输出初始化统计。
orthogonal_layer = nn.Linear(4, 4, bias=False); nn.init.orthogonal_(orthogonal_layer.weight)  # 正交初始化使方阵W乘WT接近单位阵。
assert torch.allclose(orthogonal_layer.weight @ orthogonal_layer.weight.T, torch.eye(4), atol=1e-5)  # 验证正交性。
