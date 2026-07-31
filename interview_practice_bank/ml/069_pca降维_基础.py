"""
题目 069：PCA降维_基础

要求：完成“PCA降维”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建二维样本。
2. PCA前必须中心化。
3. 计算协方差矩阵。
4. 对称矩阵特征分解。
5. 按解释方差降序。
6. 选取第一个主成分用于变式1。
7. 投影到低维空间。
8. 计算解释方差比。

完成标准：
- 验证降维shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 3.0]])  # 创建二维样本。
centered = samples - samples.mean(axis=0)  # PCA前必须中心化。
covariance = centered.T @ centered / (len(samples) - 1)  # 计算协方差矩阵。
eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # 对称矩阵特征分解。
order = eigenvalues.argsort()[::-1]  # 按解释方差降序。
components = eigenvectors[:, order[:1]]  # 选取第一个主成分用于变式1。
projected = centered @ components  # 投影到低维空间。
explained_ratio = eigenvalues[order] / eigenvalues.sum()  # 计算解释方差比。
assert projected.shape == (4, 1)  # 验证降维shape。
print(components, projected, explained_ratio)  # 输出PCA结果。
