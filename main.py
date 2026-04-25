# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: NumPy KNN.\"\"\"  # 说明这道题是手写 KNN。

import numpy as np  # 导入 NumPy，用来做距离计算和多数投票。

x_train = np.array([[1.0, 1.0], [1.0, 2.0], [4.0, 4.0], [5.0, 5.0]], dtype=np.float32)  # 定义训练特征矩阵，shape = (4, 2)。
y_train = np.array([0, 0, 1, 1], dtype=np.int32)  # 定义训练标签向量，shape = (4,)。
x_test = np.array([[2.0, 2.0]], dtype=np.float32)  # 定义测试样本矩阵，shape = (1, 2)。
k = 3  # 定义最近邻个数 k = 3。
distances = np.linalg.norm(x_train - x_test[0], axis=1)  # 计算测试样本到所有训练样本的距离，shape = (4,)。
nearest_indices = np.argsort(distances)[:k]  # 取距离最小的 k 个样本下标，shape = (3,)。
nearest_labels = y_train[nearest_indices]  # 取对应标签向量，shape = (3,)。
pred = np.bincount(nearest_labels).argmax()  # 用多数投票得到预测类别，输出是标量。

print(\"Distance shape:\", distances.shape)  # 打印距离向量 shape。
print(\"Nearest labels:\", nearest_labels)  # 打印最近邻标签。
print(\"Predicted class:\", int(pred))  # 打印最终预测类别。
