# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便中文注释正常显示。
\"\"\"Core interview problem: NumPy linear regression.\"\"\"  # 用一句话说明这道题是手写线性回归。

import numpy as np  # 导入 NumPy，用来做矩阵运算和梯度下降；这里没有张量 shape。

x_train = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], dtype=np.float32)  # 定义训练特征矩阵，shape = (3, 2)。
y_train = np.array([3.0, 3.0, 7.0], dtype=np.float32)  # 定义训练标签向量，shape = (3,)。
x_test = np.array([[4.0, 3.0]], dtype=np.float32)  # 定义测试特征矩阵，shape = (1, 2)。
y_test = np.array([7.0], dtype=np.float32)  # 定义测试标签向量，shape = (1,)。

w = np.zeros(x_train.shape[1], dtype=np.float32)  # 初始化权重向量，shape = (2,)。
b = np.float32(0.0)  # 初始化偏置标量，shape 可以看成 ()。
learning_rate = 0.1  # 定义学习率，是一个标量。
epochs = 200  # 定义训练轮数，是一个整数。
n = x_train.shape[0]  # 记录训练样本数 n = 3，是一个整数。

for epoch in range(epochs):  # 开始 full-batch gradient descent 训练循环。
    pred = x_train @ w + b  # 计算训练集预测值，pred 的 shape = (3,)。
    error = pred - y_train  # 计算预测误差，error 的 shape = (3,)。
    loss = np.mean(error ** 2)  # 计算均方误差 MSE，输出是标量。
    grad_w = (2.0 / n) * (x_train.T @ error)  # 计算损失对权重的梯度，grad_w 的 shape = (2,)。
    grad_b = (2.0 / n) * np.sum(error)  # 计算损失对偏置的梯度，grad_b 是标量。
    w = w - learning_rate * grad_w  # 按梯度下降公式更新权重，shape 仍然是 (2,)。
    b = b - learning_rate * grad_b  # 按梯度下降公式更新偏置，仍然是标量。
    if epoch % 50 == 0:  # 每 50 轮打印一次训练日志。
        print(f\"Epoch {epoch:03d} | Train MSE: {loss:.4f}\")  # 打印当前轮数和训练损失。

pred_test = x_test @ w + b  # 用训练好的参数在测试集上做预测，pred_test 的 shape = (1,)。
test_error = pred_test - y_test  # 计算测试误差，test_error 的 shape = (1,)。
test_mse = np.mean(test_error ** 2)  # 计算测试集 MSE，输出是标量。

print(\"Weight shape:\", w.shape)  # 打印权重向量的 shape。
print(\"Prediction shape:\", pred_test.shape)  # 打印预测结果的 shape。
print(\"Test MSE:\", float(test_mse))  # 打印测试集 MSE。
