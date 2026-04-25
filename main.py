# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: NumPy layer normalization.\"\"\"  # 说明这道题是手写 LayerNorm。

import numpy as np  # 导入 NumPy，用来做均值方差和标准化计算。

x = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]], dtype=np.float32)  # 定义输入矩阵，shape = (2, 3)。
gamma = np.ones((1, 3), dtype=np.float32)  # 定义缩放参数 gamma，shape = (1, 3)。
beta = np.zeros((1, 3), dtype=np.float32)  # 定义平移参数 beta，shape = (1, 3)。
eps = 1e-5  # 定义数值稳定项 eps，是标量。
mean = np.mean(x, axis=1, keepdims=True)  # 对每个样本自己的特征维求均值，mean 的 shape = (2, 1)。
var = np.var(x, axis=1, keepdims=True)  # 对每个样本自己的特征维求方差，var 的 shape = (2, 1)。
x_hat = (x - mean) / np.sqrt(var + eps)  # 按样本做标准化，x_hat 的 shape = (2, 3)。
out = gamma * x_hat + beta  # 做仿射变换得到输出，out 的 shape = (2, 3)。

print(\"Mean shape:\", mean.shape)  # 打印均值张量的 shape。
print(\"Variance shape:\", var.shape)  # 打印方差张量的 shape。
print(\"Output shape:\", out.shape)  # 打印输出张量的 shape。
