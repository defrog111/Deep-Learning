"""
题目 071：PCA降维_易错点

要求：完成“PCA降维”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建二维样本。
2. PCA前必须中心化。
3. 计算协方差矩阵。
4. 对称矩阵特征分解。
5. 按解释方差降序。
6. 选取第一个主成分用于变式3。
7. 投影到低维空间。
8. 计算解释方差比。
9. 在全数据上中心化会把测试分布泄漏给PCA。

完成标准：
- 验证降维shape。
- 验证无监督预处理同样会泄漏。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[2.0, 0.0], [0.0, 2.0], [3.0, 1.0], [1.0, 3.0]])  # 创建二维样本。
centered = samples - samples.mean(axis=0)  # PCA前必须中心化。
covariance = centered.T @ centered / (len(samples) - 1)  # 计算协方差矩阵。
eigenvalues, eigenvectors = np.linalg.eigh(covariance)  # 对称矩阵特征分解。
order = eigenvalues.argsort()[::-1]  # 按解释方差降序。
components = eigenvectors[:, order[:1]]  # 选取第一个主成分用于变式3。
projected = centered @ components  # 投影到低维空间。
explained_ratio = eigenvalues[order] / eigenvalues.sum()  # 计算解释方差比。
assert projected.shape == (4, 1)  # 验证降维shape。
print(components, projected, explained_ratio)  # 输出PCA结果。
pca_train = np.array([[0.0, 0.0], [1.0, 1.0]]); pca_test = np.array([[100.0, 100.0]]); correct_center = pca_train.mean(0); leaked_center = np.r_[pca_train, pca_test].mean(0)  # 在全数据上中心化会把测试分布泄漏给PCA。
assert not np.allclose(correct_center, leaked_center)  # 验证无监督预处理同样会泄漏。
