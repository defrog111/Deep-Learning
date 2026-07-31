"""
题目 099：指标计算与混淆矩阵_易错点

要求：完成“指标计算与混淆矩阵”的易错点题，说明训练态、梯度和张量shape。
先自己实现，再运行本文件查看参考代码结果。
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
