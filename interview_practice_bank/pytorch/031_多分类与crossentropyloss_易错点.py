"""
题目 031：多分类与CrossEntropyLoss_易错点

要求：完成“多分类与CrossEntropyLoss”的易错点题，说明训练态、梯度和张量shape。

操作步骤：
1. CrossEntropy接收未归一化logits。
2. 多分类标签使用Long类别索引。
3. 内部组合log_softmax和NLLLoss。
4. 计算logits梯度。
5. 仅展示时转换概率。
6. CrossEntropyLoss要求原始logits，提前softmax会改变梯度。

完成标准：
- 验证每行概率和为1。
- 验证常见双重softmax错误。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
from torch import nn  # 导入损失函数模块。
logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]], requires_grad=True)  # CrossEntropy接收未归一化logits。
labels = torch.tensor([0, 1])  # 多分类标签使用Long类别索引。
loss = nn.CrossEntropyLoss(label_smoothing=0.06)(logits, labels)  # 内部组合log_softmax和NLLLoss。
loss.backward()  # 计算logits梯度。
probabilities = logits.detach().softmax(dim=1)  # 仅展示时转换概率。
assert torch.allclose(probabilities.sum(1), torch.ones(2))  # 验证每行概率和为1。
print(loss.item(), probabilities, logits.grad)  # 输出损失、概率和梯度。
raw_logits = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True); correct_ce = nn.CrossEntropyLoss()(raw_logits, torch.tensor([2])); wrong_ce = nn.CrossEntropyLoss()(torch.softmax(raw_logits, dim=1), torch.tensor([2]))  # CrossEntropyLoss要求原始logits，提前softmax会改变梯度。
assert not torch.allclose(correct_ce, wrong_ce)  # 验证常见双重softmax错误。
