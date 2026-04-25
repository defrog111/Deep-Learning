# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: NumPy PCA.\"\"\"  # 说明这是手写 PCA。

import numpy as np  # 导入 NumPy，用来做协方差和特征分解。

x = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]], dtype=np.float32)  # 定义原始数据矩阵，shape = (5, 2)。
mean = np.mean(x, axis=0)  # 计算每个特征的均值向量，shape = (2,)。
x_centered = x - mean  # 对数据做中心化，shape = (5, 2)。
cov = np.cov(x_centered.T)  # 计算协方差矩阵，shape = (2, 2)。
eigenvalues, eigenvectors = np.linalg.eigh(cov)  # 做特征分解，eigenvalues 的 shape = (2,)，eigenvectors 的 shape = (2, 2)。
order = np.argsort(eigenvalues)[::-1]  # 对特征值做从大到小排序，shape = (2,)。
principal_components = eigenvectors[:, order]  # 取排序后的特征向量矩阵，shape = (2, 2)。
projection_matrix = principal_components[:, :1]  # 取前 1 个主成分作为投影矩阵，shape = (2, 1)。
x_reduced = x_centered @ projection_matrix  # 把原始数据投影到 1 维，shape = (5, 1)。

print(\"Covariance shape:\", cov.shape)  # 打印协方差矩阵的 shape。
print(\"Projection shape:\", projection_matrix.shape)  # 打印投影矩阵的 shape。
print(\"Reduced shape:\", x_reduced.shape)  # 打印降维结果的 shape。
