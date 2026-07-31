"""
题目 070：SVD奇异值分解_变式

要求：完成“SVD奇异值分解”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建非方阵。
2. 计算紧凑SVD。
3. 用UΣVᵀ重建矩阵。
4. 构造rank-1低秩近似。

完成标准：
- 验证完整SVD可重建。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]])  # 创建非方阵。
u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)  # 计算紧凑SVD。
reconstructed = u @ np.diag(singular_values) @ vt  # 用UΣVᵀ重建矩阵。
rank_k = u[:, :1] @ np.diag(singular_values[:1]) @ vt[:1]  # 构造rank-1低秩近似。
assert np.allclose(reconstructed, matrix)  # 验证完整SVD可重建。
print(singular_values, '\n', rank_k)  # 输出奇异值和低秩近似。
