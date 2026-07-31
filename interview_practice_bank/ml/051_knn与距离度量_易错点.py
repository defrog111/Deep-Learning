"""
题目 051：KNN与距离度量_易错点

要求：完成“KNN与距离度量”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建二维训练点。
2. 创建类别标签。
3. 创建待预测样本。
4. 计算欧氏距离。
5. KNN分类常使用奇数k减少平票。
6. 找到最近k个样本。
7. 多数投票。
8. 距离加权KNN让近邻贡献更大。
9. 一个很近的正类可超过两个远负类。

完成标准：
- 验证合法类别。
- 一个很近的正类可超过两个远负类。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
neighbor_distances = np.array([0.1, 0.5, 1.0]); neighbor_labels = np.array([1, 0, 0]); inverse_weights = 1 / np.maximum(neighbor_distances, 1e-12); weighted_vote = np.bincount(neighbor_labels, weights=inverse_weights).argmax()  # 距离加权KNN让近邻贡献更大。
assert weighted_vote == 1  # 一个很近的正类可超过两个远负类。
