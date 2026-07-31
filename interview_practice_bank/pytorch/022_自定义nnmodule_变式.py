"""
题目 022：自定义nnModule_变式

要求：完成“自定义nnModule”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 定义继承nn.Module的模型。
2. 初始化可训练层。
3. 注册父类内部状态。
4. 组合两层MLP。
5. 定义前向计算。
6. 返回三分类logits。
7. 实例化模型。
8. 输入随变式变化的batch。

完成标准：
- 验证最后一维是类别数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入神经网络模块。
class TinyMLP(nn.Module):  # 定义继承nn.Module的模型。
    def __init__(self):  # 初始化可训练层。
        super().__init__()  # 注册父类内部状态。
        self.network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))  # 组合两层MLP。
    def forward(self, features):  # 定义前向计算。
        return self.network(features)  # 返回三分类logits。
model = TinyMLP()  # 实例化模型。
output = model(torch.randn(4, 4))  # 输入随变式变化的batch。
assert output.shape == (output.size(0), 3)  # 验证最后一维是类别数。
print(model, output.shape, sum(parameter.numel() for parameter in model.parameters()))  # 输出结构、shape和参数量。
