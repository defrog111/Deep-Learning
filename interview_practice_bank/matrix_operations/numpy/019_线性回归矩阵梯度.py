"""
题目 019：NumPy 线性回归矩阵梯度

要求：独立完成本题，运行代码，并能口头解释每一个矩阵 shape 为什么成立。

操作步骤：
1. 用矩阵形式计算预测和 MSE。
2. 手推梯度 2/n X 转置(Xw-y)。
3. 用有限差分检查一个梯度分量。

完成标准：
- 所有 assert 通过。
- 能说明使用的 API、输入输出 shape、数值稳定性或梯度含义。
- 脚本可以独立运行。

练习方式：先遮住参考代码自己实现，再逐行对照下面的中文注释。
"""

import numpy as np  # 导入 NumPy。
x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])  # 第一列作为截距特征。
y = np.array([1.0, 3.0, 5.0])  # 创建线性目标。
weights = np.array([0.5, 1.5])  # 创建待优化参数。
residual = x @ weights - y  # 使用矩阵乘法计算预测残差。
loss = np.mean(residual**2)  # 计算均方误差。
gradient = (2.0 / len(x)) * x.T @ residual  # 手推 MSE 对权重的矩阵梯度。
epsilon = 1e-6  # 设置中心有限差分步长。
direction = np.array([1.0, 0.0])  # 只扰动第一个参数。
loss_plus = np.mean((x @ (weights + epsilon * direction) - y) ** 2)  # 计算正向扰动损失。
loss_minus = np.mean((x @ (weights - epsilon * direction) - y) ** 2)  # 计算反向扰动损失。
numeric_gradient = (loss_plus - loss_minus) / (2 * epsilon)  # 中心差分近似偏导。
assert np.isclose(gradient[0], numeric_gradient, atol=1e-6)  # 核对解析梯度。
print(loss, gradient, numeric_gradient)  # 输出损失和两类梯度。
