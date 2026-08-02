"""
题目 072：PCA降维_综合

要求：完成“PCA降维”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合SVD投影并whitening。
3. 验证白化后单位协方差。

完成标准：
- 验证白化后单位协方差。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
pca_matrix = np.array([[2.0, 0.0], [0.0, 1.0], [-2.0, 0.0], [0.0, -1.0]]); pca_centered = pca_matrix - pca_matrix.mean(0); _, pca_singular, pca_vt = np.linalg.svd(pca_centered, full_matrices=False); whitened = (pca_centered @ pca_vt.T) / (pca_singular / np.sqrt(len(pca_matrix) - 1))  # 综合SVD投影并whitening。
assert np.allclose(np.cov(whitened, rowvar=False), np.eye(2))  # 验证白化后单位协方差。
