"""
题目 052：KNN与距离度量_综合

要求：完成“KNN与距离度量”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 高维中最近和最远距离趋于相似。
3. 综合演示维度灾难。

完成标准：
- 综合演示维度灾难。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng_dimension = np.random.default_rng(0); low_dim = rng_dimension.random((1000, 2)); high_dim = rng_dimension.random((1000, 50)); low_ratio = np.linalg.norm(low_dim, axis=1).min() / np.linalg.norm(low_dim, axis=1).max(); high_ratio = np.linalg.norm(high_dim, axis=1).min() / np.linalg.norm(high_dim, axis=1).max()  # 高维中最近和最远距离趋于相似。
assert high_ratio > low_ratio  # 综合演示维度灾难。
