# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: standardization and normalization.\"\"\"  # 说明这道题是手写两种预处理。

import numpy as np  # 导入 NumPy，用来做均值、标准差和最值计算。

x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)  # 定义特征矩阵，shape = (3, 2)。
mean = np.mean(x, axis=0, keepdims=True)  # 计算每列均值，mean 的 shape = (1, 2)。
std = np.std(x, axis=0, keepdims=True)  # 计算每列标准差，std 的 shape = (1, 2)。
x_standardized = (x - mean) / std  # 做 z-score 标准化，shape = (3, 2)。
min_value = np.min(x, axis=0, keepdims=True)  # 计算每列最小值，shape = (1, 2)。
max_value = np.max(x, axis=0, keepdims=True)  # 计算每列最大值，shape = (1, 2)。
x_normalized = (x - min_value) / (max_value - min_value)  # 做 min-max 归一化，shape = (3, 2)。

print(\"Standardized shape:\", x_standardized.shape)  # 打印标准化结果 shape。
print(\"Normalized shape:\", x_normalized.shape)  # 打印归一化结果 shape。
