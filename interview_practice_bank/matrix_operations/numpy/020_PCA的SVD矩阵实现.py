"""
题目 020：NumPy PCA的SVD矩阵实现

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 中心化二维数据。
2. 用 SVD 找主方向。
3. 投影到一维并重建，解释符号不唯一。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
samples = np.array([[1.0, 1.0], [2.0, 2.1], [3.0, 2.9], [4.0, 4.2]])  # 创建强相关二维数据。
mean = samples.mean(axis=0, keepdims=True)  # 计算每个特征均值。
centered = samples - mean  # PCA 前先中心化。
_, singular, vh = np.linalg.svd(centered, full_matrices=False)  # 右奇异向量给出主轴。
component = vh[:1]  # 选择解释方差最大的第一主成分。
scores = centered @ component.T  # 把样本投影到一维主成分空间。
reconstructed = scores @ component + mean  # 从一维分数近似重建二维数据。
explained_ratio = singular[0] ** 2 / np.sum(singular**2)  # 计算第一主成分解释方差比例。
assert scores.shape == (4, 1) and explained_ratio > 0.99  # 验证降维 shape 和解释率。
print(component, scores, reconstructed, explained_ratio)  # 输出 PCA 中间结果。
