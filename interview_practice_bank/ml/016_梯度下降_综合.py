"""
题目 016：梯度下降_综合

要求：完成“梯度下降”的综合题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
x = np.array([1.0, 2.0, 3.0])  # 创建训练特征。
y = 2 * x + 1  # 创建线性目标。
weight, bias = 0.0, 0.0  # 初始化模型参数。
learning_rate = 0.07  # 设置学习率。
for _ in range(500):  # 迭代梯度下降。
    errors = weight * x + bias - y  # 计算预测残差。
    weight -= learning_rate * 2 * np.mean(errors * x)  # 更新权重。
    bias -= learning_rate * 2 * np.mean(errors)  # 更新偏置。
assert abs(weight - 2) < 0.05 and abs(bias - 1) < 0.1  # 验证收敛到真实参数。
print(weight, bias)  # 输出优化结果。
