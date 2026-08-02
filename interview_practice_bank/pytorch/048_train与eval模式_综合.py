"""
题目 048：train与eval模式_综合

要求：完成“train与eval模式”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. eval递归切换所有子模块模式。
2. 推理还需单独关闭梯度。
    mode_prediction = mode_model(torch.ones(1, 2))  # 执行确定性推理。
3. 综合验证模式与梯度。

完成标准：
- 综合验证模式与梯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入模型模块。
mode_model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout()); mode_model.eval()  # eval递归切换所有子模块模式。
with torch.inference_mode():  # 推理还需单独关闭梯度。
    mode_prediction = mode_model(torch.ones(1, 2))  # 执行确定性推理。
assert not mode_model.training and not mode_model[0].training and not mode_prediction.requires_grad  # 综合验证模式与梯度。
