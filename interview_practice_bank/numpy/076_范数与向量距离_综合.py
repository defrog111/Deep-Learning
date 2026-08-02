"""
题目 076：范数与向量距离_综合

要求：完成“范数与向量距离”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建向量a。
2. 创建向量b。
3. 综合计算余弦相似度。
4. 非零向量与自身余弦相似度为1。

完成标准：
- 非零向量与自身余弦相似度为1。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
a = np.array([1.0, 2.0, 3.0])  # 创建向量a。
b = np.array([4.0, 2.0, 0.0])  # 创建向量b。
norm_matrix = np.stack([a, b]); cosine_matrix = norm_matrix @ norm_matrix.T / np.maximum(np.linalg.norm(norm_matrix, axis=1)[:, None] * np.linalg.norm(norm_matrix, axis=1)[None, :], 1e-12)  # 综合计算余弦相似度。
assert np.allclose(np.diag(cosine_matrix), 1)  # 非零向量与自身余弦相似度为1。
