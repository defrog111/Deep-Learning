"""
题目 070：SVD奇异值分解_变式

要求：完成“SVD奇异值分解”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. 创建非方阵。
3. 经济型SVD减少无用维度。
4. 验证紧凑SVD shape。

完成标准：
- 验证紧凑SVD shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]])  # 创建非方阵。
compact_u, compact_s, compact_vt = np.linalg.svd(matrix, full_matrices=False)  # 经济型SVD减少无用维度。
assert compact_u.shape[1] == compact_s.size == compact_vt.shape[0]  # 验证紧凑SVD shape。
