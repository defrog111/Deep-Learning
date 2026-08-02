"""
题目 063：SVM间隔与hinge loss_易错点

要求：完成“SVM间隔与hinge loss”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 核技巧只通过样本相似度隐式映射。
3. 验证两种Gram矩阵。

完成标准：
- 验证两种Gram矩阵。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
kernel_points = np.array([[1.0, 2.0], [2.0, 1.0]]); linear_kernel = kernel_points @ kernel_points.T; gamma = 0.5; squared_distances = ((kernel_points[:, None] - kernel_points[None, :])**2).sum(-1); rbf_kernel = np.exp(-gamma * squared_distances)  # 核技巧只通过样本相似度隐式映射。
assert np.allclose(np.diag(rbf_kernel), 1) and linear_kernel.shape == rbf_kernel.shape  # 验证两种Gram矩阵。
