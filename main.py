# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: train val test split.\"\"\"  # 说明这道题是手写数据切分。

import numpy as np  # 导入 NumPy，用来做打乱和切片。

x = np.arange(30, dtype=np.float32).reshape(10, 3)  # 定义 10 条样本的特征矩阵，shape = (10, 3)。
y = np.arange(10, dtype=np.int32)  # 定义对应标签向量，shape = (10,)。
indices = np.arange(len(x))  # 生成样本下标向量，shape = (10,)。
rng = np.random.default_rng(seed=42)  # 创建随机数生成器，方便可复现打乱。
rng.shuffle(indices)  # 原地打乱下标向量，shape 仍然是 (10,)。
x = x[indices]  # 按打乱后的下标重排特征矩阵，shape 仍然是 (10, 3)。
y = y[indices]  # 按打乱后的下标重排标签向量，shape 仍然是 (10,)。
train_end = 6  # 定义训练集结束位置，对应前 60% 数据。
val_end = 8  # 定义验证集结束位置，对应再取 20% 数据。
x_train = x[:train_end]  # 切出训练特征矩阵，shape = (6, 3)。
y_train = y[:train_end]  # 切出训练标签向量，shape = (6,)。
x_val = x[train_end:val_end]  # 切出验证特征矩阵，shape = (2, 3)。
y_val = y[train_end:val_end]  # 切出验证标签向量，shape = (2,)。
x_test = x[val_end:]  # 切出测试特征矩阵，shape = (2, 3)。
y_test = y[val_end:]  # 切出测试标签向量，shape = (2,)。

print(\"Train shape:\", x_train.shape, y_train.shape)  # 打印训练集特征和标签的 shape。
print(\"Val shape:\", x_val.shape, y_val.shape)  # 打印验证集特征和标签的 shape。
print(\"Test shape:\", x_test.shape, y_test.shape)  # 打印测试集特征和标签的 shape。
