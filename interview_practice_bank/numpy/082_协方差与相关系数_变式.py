"""
题目 082：协方差与相关系数_变式

要求：完成“协方差与相关系数”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 8.0]])  # 每行一个样本每列一个特征。
covariance = np.cov(samples, rowvar=False, ddof=1)  # 计算样本协方差矩阵。
correlation = np.corrcoef(samples, rowvar=False)  # 计算标准化相关系数。
centered = samples - samples.mean(axis=0, keepdims=True)  # 对特征中心化。
manual_covariance = centered.T @ centered / (len(samples) - 1)  # 手算样本协方差。
assert np.allclose(covariance, manual_covariance)  # 验证公式实现。
print(covariance, '\n', correlation)  # 输出协方差和相关矩阵。
