"""
题目 021：MSEMAE与Huber损失_基础

要求：完成“MSEMAE与Huber损失”的基础题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
targets = np.array([1.0, 2.0, 3.0, 20.0])  # 构造含离群点目标。
predictions = np.array([1.2, 1.8, 3.1, 5.0])  # 构造预测。
errors = predictions - targets  # 计算残差。
mse = np.mean(errors**2)  # MSE对大误差平方惩罚。
mae = np.mean(np.abs(errors))  # MAE对离群点更稳健。
delta = 3.0  # 设置Huber转折点。
huber = np.mean(np.where(np.abs(errors) <= delta, 0.5 * errors**2, delta * (np.abs(errors) - 0.5 * delta)))  # 分段计算Huber损失。
assert mse > mae  # 本例离群点使MSE明显更大。
print(mse, mae, huber)  # 输出三种回归损失。
