"""
题目 076：特征选择与互信息思想_综合

要求：完成“特征选择与互信息思想”的综合题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
features = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]])  # 创建三个二值特征。
target = np.array([0, 0, 1, 1])  # 第一个特征完全决定标签。
correlations = np.array([np.corrcoef(features[:, index], target)[0, 1] for index in range(features.shape[1])])  # 用相关性演示单变量筛选。
selected = np.abs(correlations).argmax()  # 选择绝对相关最高特征。
variance = features.var(axis=0)  # 低方差筛选与标签无关。
keep_variance = variance >= 0.4  # 使用方差阈值。
assert selected == 0  # 验证找到最直接相关特征。
print(correlations, variance, keep_variance, selected)  # 输出筛选依据。
