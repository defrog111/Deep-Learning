"""
题目 100：指标计算与混淆矩阵_综合

要求：完成“指标计算与混淆矩阵”的综合题，说明训练态、梯度和张量shape。

操作步骤：
1. 创建三分类预测。
2. 创建真实标签。
3. 设置类别数。
4. 把二维类别对编码为一维索引。
5. 构建混淆矩阵。
6. 取得各类TP。
7. 按预测列计算precision。
8. 按真实行计算recall。
9. 计算整体准确率。
10. 综合计算多标签micro-F1。

完成标准：
- 验证每个样本被统计一次。
- 验证TP、FP、FN公式。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import torch  # 导入 PyTorch。
predictions = torch.tensor([0, 1, 2, 1, 0, 2, 1, 1])  # 创建三分类预测。
targets = torch.tensor([0, 2, 2, 1, 0, 0, 1, 2])  # 创建真实标签。
num_classes = 3  # 设置类别数。
indices = targets * num_classes + predictions  # 把二维类别对编码为一维索引。
confusion = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)  # 构建混淆矩阵。
true_positive = confusion.diag().float()  # 取得各类TP。
precision = true_positive / confusion.sum(0).clamp_min(1)  # 按预测列计算precision。
recall = true_positive / confusion.sum(1).clamp_min(1)  # 按真实行计算recall。
accuracy = true_positive.sum() / confusion.sum()  # 计算整体准确率。
assert confusion.sum() == len(targets)  # 验证每个样本被统计一次。
print(confusion, precision, recall, accuracy)  # 输出分类指标。
multi_predictions = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool); multi_targets = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.bool); micro_tp = (multi_predictions & multi_targets).sum(); micro_fp = (multi_predictions & ~multi_targets).sum(); micro_fn = (~multi_predictions & multi_targets).sum(); micro_f1 = 2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn)  # 综合计算多标签micro-F1。
assert torch.isclose(micro_f1, torch.tensor(2 / 3))  # 验证TP、FP、FN公式。
