"""
题目 075：范数与向量距离_易错点

要求：完成“范数与向量距离”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 创建向量a。
2. 创建向量b。
3. 利用广播计算两两欧氏距离。

完成标准：
- 验证距离矩阵。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
a = np.array([1.0, 2.0, 3.0])  # 创建向量a。
b = np.array([4.0, 2.0, 0.0])  # 创建向量b。
norm_matrix = np.stack([a, b]); pairwise = np.linalg.norm(norm_matrix[:, None, :] - norm_matrix[None, :, :], axis=-1)  # 利用广播计算两两欧氏距离。
assert pairwise.shape == (len(norm_matrix), len(norm_matrix)) and np.allclose(np.diag(pairwise), 0)  # 验证距离矩阵。
