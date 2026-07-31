"""
题目 038：混淆矩阵与分类指标_变式

要求：完成“混淆矩阵与分类指标”的变式题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
targets = np.array([1, 1, 1, 0, 0, 0])  # 创建真实二分类标签。
predictions = np.array([1, 1, 0, 1, 0, 0])  # 创建预测。
tp = np.sum((targets == 1) & (predictions == 1))  # 计算真正例。
fp = np.sum((targets == 0) & (predictions == 1))  # 计算假正例。
fn = np.sum((targets == 1) & (predictions == 0))  # 计算假负例。
tn = np.sum((targets == 0) & (predictions == 0))  # 计算真负例。
precision = tp / (tp + fp)  # 计算查准率。
recall = tp / (tp + fn)  # 计算查全率。
f1 = 2 * precision * recall / (precision + recall)  # 计算F1调和平均。
assert tp + fp + fn + tn == len(targets)  # 验证混淆矩阵覆盖全部样本。
print(tp, fp, fn, tn, precision, recall, f1)  # 输出分类指标。
