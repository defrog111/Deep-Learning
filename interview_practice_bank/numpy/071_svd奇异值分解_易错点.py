"""
题目 071：SVD奇异值分解_易错点

要求：完成“SVD奇异值分解”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建非方阵。
2. 计算紧凑SVD。
3. 只需要奇异值时不计算左右奇异向量。

完成标准：
- 验证两种调用得到相同奇异值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]])  # 创建非方阵。
u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)  # 计算紧凑SVD。
singular_only = np.linalg.svd(matrix, compute_uv=False)  # 只需要奇异值时不计算左右奇异向量。
assert np.allclose(singular_only, singular_values)  # 验证两种调用得到相同奇异值。
