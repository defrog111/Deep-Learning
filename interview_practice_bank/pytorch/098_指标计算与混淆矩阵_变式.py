"""
题目 098：指标计算与混淆矩阵_变式

要求：完成“指标计算与混淆矩阵”的变式题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 计算Top-2准确率。
3. 验证两个目标都进入前二。

完成标准：
- 验证两个目标都进入前二。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
score_matrix = torch.tensor([[0.1, 0.8, 0.1], [0.4, 0.3, 0.3]]); top2 = score_matrix.topk(2, dim=1).indices; metric_targets = torch.tensor([1, 0]); top2_accuracy = top2.eq(metric_targets[:, None]).any(1).float().mean()  # 计算Top-2准确率。
assert top2_accuracy == 1  # 验证两个目标都进入前二。
