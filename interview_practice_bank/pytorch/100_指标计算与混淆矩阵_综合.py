"""
题目 100：指标计算与混淆矩阵_综合

要求：完成“指标计算与混淆矩阵”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 导入 PyTorch。
2. 综合计算多标签micro-F1。
3. 验证TP、FP、FN公式。

完成标准：
- 验证TP、FP、FN公式。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
multi_predictions = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool); multi_targets = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.bool); micro_tp = (multi_predictions & multi_targets).sum(); micro_fp = (multi_predictions & ~multi_targets).sum(); micro_fn = (~multi_predictions & multi_targets).sum(); micro_f1 = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)  # 综合计算多标签micro-F1。
assert torch.isclose(micro_f1, torch.tensor(2 / 3))  # 验证TP、FP、FN公式。
