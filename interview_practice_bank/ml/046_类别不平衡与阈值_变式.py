"""
题目 046：类别不平衡与阈值_变式

要求：完成“类别不平衡与阈值”的变式题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
targets = np.array([1, 1, 0, 0, 0, 0, 0, 0])  # 构造不平衡标签。
scores = np.array([0.9, 0.45, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05])  # 构造模型概率。
threshold = 0.5  # 根据业务代价调整阈值。
predictions = scores >= threshold  # 把概率转类别。
recall = np.sum(predictions & (targets == 1)) / np.sum(targets == 1)  # 计算少数类召回率。
balanced_weight = len(targets) / (2 * np.bincount(targets))  # 计算常见balanced类别权重。
assert len(balanced_weight) == 2  # 验证每类一个权重。
print(threshold, predictions, recall, balanced_weight)  # 输出阈值和不平衡处理信息。
