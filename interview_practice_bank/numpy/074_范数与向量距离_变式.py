"""
题目 074：范数与向量距离_变式

要求：完成“范数与向量距离”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建向量a。
2. 创建向量b。
3. 沿行计算范数并防止除零。

完成标准：
- 验证行归一化。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
a = np.array([1.0, 2.0, 3.0])  # 创建向量a。
b = np.array([4.0, 2.0, 0.0])  # 创建向量b。
norm_matrix = np.stack([a, b]); row_norms_extra = np.linalg.norm(norm_matrix, axis=1, keepdims=True); normalized_rows = norm_matrix / np.maximum(row_norms_extra, 1e-12)  # 沿行计算范数并防止除零。
assert np.allclose(np.linalg.norm(normalized_rows, axis=1), 1)  # 验证行归一化。
