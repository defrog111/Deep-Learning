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

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
explained_variance_ratio = eigenvalues[order] / np.sum(eigenvalues)  # è®¡ç®—æ¯ä¸ªä¸»æˆåˆ†çš„æ–¹å·®è§£é‡ŠçŽ‡ï¼Œshape = (2,)ã€‚
reconstructed = x_reduced @ projection_matrix.T + mean  # æŠŠé™ç»´ç»“æžœæŠ•å½±å›žåŽŸç©ºé—´ï¼Œshape = (5, 2)ã€‚
reconstruction_error = np.mean((x - reconstructed) ** 2)  # è®¡ç®—é‡å»ºè¯¯å·®ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Explained variance ratio:", explained_variance_ratio)  # æ‰“å°æ–¹å·®è§£é‡ŠçŽ‡ã€‚
print("Reconstruction error:", float(reconstruction_error))  # æ‰“å°é‡å»ºè¯¯å·®ã€‚
