"""
题目 016：梯度下降_综合

要求：完成“梯度下降”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建训练特征。
2. 创建线性目标。
3. 初始化模型参数。
4. 设置学习率。
5. 迭代梯度下降。
6. 计算预测残差。
7. 更新权重。
8. 更新偏置。

完成标准：
- 验证收敛到真实参数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
