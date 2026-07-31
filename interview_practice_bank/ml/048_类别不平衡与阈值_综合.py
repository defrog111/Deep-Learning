"""
题目 048：类别不平衡与阈值_综合

要求：完成“类别不平衡与阈值”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造不平衡标签。
2. 构造模型概率。
3. 根据业务代价调整阈值。
4. 把概率转类别。
5. 计算少数类召回率。
6. 计算常见balanced类别权重。
7. balanced权重与类别频次成反比。
8. 综合验证加权守恒。

完成标准：
- 验证每类一个权重。
- 综合验证加权守恒。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
targets = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # 构造不平衡标签。
scores = np.array([0.9, 0.45, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])  # 构造模型概率。
threshold = 0.7  # 根据业务代价调整阈值。
predictions = scores >= threshold  # 把概率转类别。
recall = np.sum(predictions & (targets == 1)) / np.sum(targets == 1)  # 计算少数类召回率。
balanced_weight = len(targets) / (2 * np.bincount(targets))  # 计算常见balanced类别权重。
assert len(balanced_weight) == 2  # 验证每类一个权重。
print(threshold, predictions, recall, balanced_weight)  # 输出阈值和不平衡处理信息。
class_counts = np.array([90, 10]); balanced_weights = class_counts.sum() / (len(class_counts) * class_counts); sample_labels = np.array([0, 0, 1]); sample_weights = balanced_weights[sample_labels]  # balanced权重与类别频次成反比。
assert sample_weights[-1] > sample_weights[0] and np.isclose((balanced_weights * class_counts).sum(), class_counts.sum())  # 综合验证加权守恒。
