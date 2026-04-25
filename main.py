# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: hinge loss.\"\"\"  # 说明这道题是手写 hinge loss。

import numpy as np  # 导入 NumPy，用来做向量运算。

x = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -1.5]], dtype=np.float32)  # 定义样本矩阵，shape = (3, 2)。
y = np.array([1.0, 1.0, -1.0], dtype=np.float32)  # 定义标签向量，取值为 -1 或 1，shape = (3,)。
w = np.array([0.5, 0.5], dtype=np.float32)  # 定义权重向量，shape = (2,)。
b = np.float32(0.0)  # 定义偏置标量。
scores = x @ w + b  # 计算线性分数向量，shape = (3,)。
margins = 1.0 - y * scores  # 计算每个样本的 margin，shape = (3,)。
losses = np.maximum(0.0, margins)  # 对每个 margin 取 max(0, margin)，shape = (3,)。
loss = np.mean(losses)  # 对所有样本的 hinge loss 求平均，输出是标量。

print(\"Margin shape:\", margins.shape)  # 打印 margin 向量的 shape。
print(\"Loss vector shape:\", losses.shape)  # 打印逐样本 loss 向量的 shape。
print(\"Mean hinge loss:\", float(loss))  # 打印平均 hinge loss。
