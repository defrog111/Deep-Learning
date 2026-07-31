"""
题目 003：StandardScaler只拟合训练集_基础

要求：完成“StandardScaler只拟合训练集”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
from sklearn.preprocessing import StandardScaler  # 导入标准化器。
train = np.array([[1.0], [2.0], [3.0]])  # 创建训练数据。
test = np.array([[100.0]])  # 创建分布不同的测试数据。
scaler = StandardScaler().fit(train)  # 只能在训练集拟合均值和方差。
train_scaled = scaler.transform(train)  # 使用训练统计量转换训练集。
test_scaled = scaler.transform(test)  # 使用同一统计量转换测试集。
assert np.isclose(train_scaled.mean(), 0) and test_scaled.item() > 10  # 验证没有用测试数据污染统计量。
print(scaler.mean_, scaler.scale_, test_scaled)  # 输出拟合参数和转换结果。
