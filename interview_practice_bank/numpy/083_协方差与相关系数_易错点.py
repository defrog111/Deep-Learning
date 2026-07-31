"""
题目 083：协方差与相关系数_易错点

要求：完成“协方差与相关系数”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 每行一个样本每列一个特征。
2. 计算样本协方差矩阵。
3. 计算标准化相关系数。
4. 对特征中心化。
5. 手算样本协方差。

完成标准：
- 验证公式实现。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0], [4.0, 8.0]])  # 每行一个样本每列一个特征。
covariance = np.cov(samples, rowvar=False, ddof=1)  # 计算样本协方差矩阵。
correlation = np.corrcoef(samples, rowvar=False)  # 计算标准化相关系数。
centered = samples - samples.mean(axis=0, keepdims=True)  # 对特征中心化。
manual_covariance = centered.T @ centered / (len(samples) - 1)  # 手算样本协方差。
assert np.allclose(covariance, manual_covariance)  # 验证公式实现。
print(covariance, '\n', correlation)  # 输出协方差和相关矩阵。
