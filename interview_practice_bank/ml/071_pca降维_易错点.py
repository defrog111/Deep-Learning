"""
题目 071：PCA降维_易错点

要求：完成“PCA降维”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 在全数据上中心化会把测试分布泄漏给PCA。
3. 验证无监督预处理同样会泄漏。

完成标准：
- 验证无监督预处理同样会泄漏。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
pca_train = np.array([[0.0, 0.0], [1.0, 1.0]]); pca_test = np.array([[100.0, 100.0]]); correct_center = pca_train.mean(0); leaked_center = np.r_[pca_train, pca_test].mean(0)  # 在全数据上中心化会把测试分布泄漏给PCA。
assert not np.allclose(correct_center, leaked_center)  # 验证无监督预处理同样会泄漏。
