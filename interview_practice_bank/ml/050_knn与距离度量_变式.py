"""
题目 050：KNN与距离度量_变式

要求：完成“KNN与距离度量”的变式题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
training = np.array([[0.0, 0.0], [1.0, 1.0], [4.0, 4.0], [5.0, 5.0]])  # 创建二维训练点。
labels = np.array([0, 0, 1, 1])  # 创建类别标签。
query = np.array([3.5, 3.5])  # 创建待预测样本。
distances = np.linalg.norm(training - query, axis=1)  # 计算欧氏距离。
k = 3  # KNN分类常使用奇数k减少平票。
neighbors = np.argsort(distances)[:k]  # 找到最近k个样本。
prediction = np.bincount(labels[neighbors], minlength=2).argmax()  # 多数投票。
assert prediction in (0, 1)  # 验证合法类别。
print(distances, neighbors, prediction)  # 输出邻居和预测。
