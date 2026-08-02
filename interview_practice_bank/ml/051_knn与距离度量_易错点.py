"""
题目 051：KNN与距离度量_易错点

要求：完成“KNN与距离度量”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 距离加权KNN让近邻贡献更大。
3. 一个很近的正类可超过两个远负类。

完成标准：
- 一个很近的正类可超过两个远负类。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
neighbor_distances = np.array([0.1, 0.5, 1.0]); neighbor_labels = np.array([1, 0, 0]); inverse_weights = 1 / np.maximum(neighbor_distances, 1e-12); weighted_vote = np.bincount(neighbor_labels, weights=inverse_weights).argmax()  # 距离加权KNN让近邻贡献更大。
assert weighted_vote == 1  # 一个很近的正类可超过两个远负类。
