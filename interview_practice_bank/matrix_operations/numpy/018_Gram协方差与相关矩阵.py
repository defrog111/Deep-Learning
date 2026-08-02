"""
题目 018：NumPy Gram协方差与相关矩阵

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 把特征按列中心化。
2. 用矩阵乘法手算样本协方差。
3. 转换为相关矩阵并与 corrcoef 核对。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 2.0], [2.0, 4.0], [4.0, 5.0], [5.0, 7.0]])  # 行是样本、列是特征。
centered = samples - samples.mean(axis=0, keepdims=True)  # 每列减去训练样本均值。
gram = centered.T @ centered  # 计算中心化特征的 Gram 矩阵。
covariance = gram / (len(samples) - 1)  # 使用 n-1 得到无偏样本协方差。
std = np.sqrt(np.diag(covariance))  # 从协方差对角线取得样本标准差。
correlation = covariance / np.outer(std, std)  # 标准化协方差得到相关矩阵。
assert np.allclose(covariance, np.cov(samples, rowvar=False))  # 与 np.cov 核对。
assert np.allclose(correlation, np.corrcoef(samples, rowvar=False))  # 与 np.corrcoef 核对。
print(gram, covariance, correlation)  # 输出三类矩阵。
