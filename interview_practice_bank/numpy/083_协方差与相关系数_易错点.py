"""
题目 083：协方差与相关系数_易错点

要求：完成“协方差与相关系数”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 每行一个样本每列一个特征。
2. 计算样本协方差矩阵。
3. ddof=0计算总体协方差，默认ddof=1是样本协方差。
4. 小样本下两种分母结果不同。

完成标准：
- 小样本下两种分母结果不同。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 8.0]])  # 每行一个样本每列一个特征。
covariance = np.cov(samples, rowvar=False, ddof=1)  # 计算样本协方差矩阵。
population_covariance = np.cov(samples, rowvar=False, ddof=0)  # ddof=0计算总体协方差，默认ddof=1是样本协方差。
assert not np.allclose(population_covariance, covariance)  # 小样本下两种分母结果不同。
