"""
题目 066：KMeans聚类_变式

要求：完成“KMeans聚类”的变式题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
points = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0], [5.0, 6.0]])  # 创建两个明显簇。
centers = np.array([[0.0, 0.0], [5.0, 5.0]])  # 初始化两个中心。
for _ in range(5):  # 交替执行分配和更新。
    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # 计算样本到各中心平方距离。
    assignments = distances.argmin(axis=1)  # 分配到最近中心。
    centers = np.stack([points[assignments == cluster].mean(axis=0) for cluster in range(2)])  # 更新为簇均值。
inertia = np.sum((points - centers[assignments]) ** 2)  # 计算簇内平方和。
assert inertia <= 1.0  # 验证简单数据被正确聚类。
print(assignments, centers, inertia)  # 输出聚类结果。
