# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Minimal NumPy linear regression interview version."""  # 用一句话说明这个脚本的用途。

import numpy as np  # 导入 NumPy，用来做矩阵运算和手写梯度下降。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，作为回归任务示例。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分训练集和测试集。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，让特征更容易训练收敛。


def prepare_data():  # 负责数据下载、切分、标准化和类型转换。
    data = fetch_california_housing()  # 下载或读取加州房价数据集。
    x = data.data  # 输入特征矩阵，shape = (20640, 8)。
    y = data.target  # 回归标签向量，shape = (20640,)。

    x_train, x_test, y_train, y_test = train_test_split(  # 把原始数据切成训练集和测试集。
        x,  # 全部特征，shape = (20640, 8)。
        y,  # 全部标签，shape = (20640,)。
        test_size=0.2,  # 抽 20% 作为测试集。
        random_state=42,  # 固定随机种子，保证划分稳定。
    )

    scaler = StandardScaler()  # 创建标准化器对象，只能在训练集上 fit。
    x_train = scaler.fit_transform(x_train)  # 在训练集上学习均值和方差后做标准化，shape 仍是 (num_train, 8)。
    x_test = scaler.transform(x_test)  # 用训练集的标准化规则转换测试集，shape 仍是 (num_test, 8)。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32，shape = (num_train, 8)。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32，shape = (num_test, 8)。
    y_train = y_train.astype(np.float32)  # 把训练标签转成 float32，shape = (num_train,)。
    y_test = y_test.astype(np.float32)  # 把测试标签转成 float32，shape = (num_test,)。
    return x_train, x_test, y_train, y_test  # 返回训练集和测试集。


def train_linear_regression(x_train, y_train, learning_rate=0.01, epochs=200):  # 手写线性回归训练函数。
    w = np.zeros(x_train.shape[1], dtype=np.float32)  # 初始化权重向量，shape = (8,)。
    b = np.float32(0.0)  # 初始化偏置，它是一个标量。
    n = len(x_train)  # 训练样本数，是一个整数。

    for epoch in range(epochs):  # 外层循环控制训练轮数。
        pred = x_train @ w + b  # 前向传播，预测值 shape = (num_train,)。
        error = pred - y_train  # 预测误差，shape = (num_train,)。
        loss = np.mean(error ** 2)  # 均方误差 MSE，输出是一个标量。

        grad_w = (2.0 / n) * (x_train.T @ error)  # 损失对权重的梯度，shape = (8,)。
        grad_b = (2.0 / n) * np.sum(error)  # 损失对偏置的梯度，输出是一个标量。

        w -= learning_rate * grad_w  # 按梯度下降公式更新权重。
        b -= learning_rate * grad_b  # 按梯度下降公式更新偏置。

        if epoch % 20 == 0:  # 每 20 轮打印一次，方便观察是否收敛。
            print(f"Epoch {epoch}, Train Loss: {loss:.4f}")  # 打印当前轮数和训练损失。

    return w, b  # 返回训练好的权重和偏置。


def evaluate(x_data, y_data, w, b):  # 负责在测试集上做推理和指标计算。
    pred = x_data @ w + b  # 用训练好的参数做预测，pred shape = (num_samples,)。
    mse = np.mean((pred - y_data) ** 2)  # 计算均方误差，输出是标量。
    mae = np.mean(np.abs(pred - y_data))  # 计算平均绝对误差，输出是标量。
    total_var = np.sum((y_data - np.mean(y_data)) ** 2)  # 计算总平方和 SST，输出是标量。
    residual_var = np.sum((y_data - pred) ** 2)  # 计算残差平方和 SSR，输出是标量。
    r2 = 1 - residual_var / total_var  # 根据 R2 = 1 - SSR / SST 计算拟合优度。
    return pred, float(mse), float(mae), float(r2)  # 返回预测值和关键指标。


def main():  # 主入口函数，把完整流程串起来。
    x_train, x_test, y_train, y_test = prepare_data()  # 准备训练集和测试集。
    w, b = train_linear_regression(x_train, y_train)  # 训练手写线性回归模型。
    pred, mse, mae, r2 = evaluate(x_test, y_test, w, b)  # 在测试集上做推理和评估。

    print(f"\nWeight shape: {w.shape}")  # 打印权重向量 shape，当前一般是 (8,)。
    print("Bias shape: scalar")  # 打印偏置是标量，不是向量。
    print(f"Prediction shape: {pred.shape}")  # 打印预测结果 shape，当前一般是 (num_test,)。
    print("Test MSE:", mse)  # 打印测试集均方误差。
    print("Test MAE:", mae)  # 打印测试集平均绝对误差。
    print("Test R2:", r2)  # 打印测试集 R2。


if __name__ == "__main__":  # 直接运行这个文件时才执行 main，被别的文件导入时不会自动运行。
    main()
