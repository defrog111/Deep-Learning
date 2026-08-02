"""
题目 066：KMeans聚类_变式

要求：完成“KMeans聚类”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. KMeans++按到最近中心距离平方选新中心。
3. 验证概率初始化。

完成标准：
- 验证概率初始化。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng_kpp = np.random.default_rng(2); points_kpp = np.array([[0.0], [1.0], [9.0], [10.0]]); first_center = points_kpp[0]; squared_nearest = ((points_kpp - first_center)**2).ravel(); second_center = points_kpp[rng_kpp.choice(len(points_kpp), p=squared_nearest / squared_nearest.sum())]  # KMeans++按到最近中心距离平方选新中心。
assert second_center.shape == (1,)  # 验证概率初始化。
