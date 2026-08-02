"""
题目 072：SVD奇异值分解_综合

要求：完成“SVD奇异值分解”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建非方阵。
2. 计算紧凑SVD。
3. 按累计能量选择低秩维数。

完成标准：
- 验证自动选择的秩范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
matrix = np.array([[3.0, 1.0], [1.0, 3.0], [1.0, 1.0]])  # 创建非方阵。
u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)  # 计算紧凑SVD。
energy_ratio = np.cumsum(singular_values**2) / np.sum(singular_values**2); rank_for_90 = np.searchsorted(energy_ratio, 0.9) + 1  # 按累计能量选择低秩维数。
assert 1 <= rank_for_90 <= singular_values.size  # 验证自动选择的秩范围。
