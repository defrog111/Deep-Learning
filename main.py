# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: compare SGD and Adam updates.\"\"\"  # 说明这是对比 SGD 和 Adam 的题目。

import numpy as np  # 导入 NumPy，用来模拟参数和梯度更新。

param = np.array([1.0, -2.0], dtype=np.float32)  # 定义初始参数向量，shape = (2,)。
grad = np.array([0.3, -0.1], dtype=np.float32)  # 定义当前梯度向量，shape = (2,)。
learning_rate = 0.1  # 定义学习率，是标量。

sgd_param = param.copy()  # 复制参数给 SGD 使用，shape = (2,)。
sgd_param = sgd_param - learning_rate * grad  # 按 SGD 规则更新参数。

adam_param = param.copy()  # 复制参数给 Adam 使用，shape = (2,)。
m = np.zeros_like(adam_param)  # 初始化一阶动量向量 m，shape = (2,)。
v = np.zeros_like(adam_param)  # 初始化二阶动量向量 v，shape = (2,)。
beta1 = 0.9  # 定义一阶动量衰减系数。
beta2 = 0.999  # 定义二阶动量衰减系数。
eps = 1e-8  # 定义数值稳定项。
t = 1  # 定义当前时间步。
m = beta1 * m + (1.0 - beta1) * grad  # 更新一阶动量，shape = (2,)。
v = beta2 * v + (1.0 - beta2) * (grad ** 2)  # 更新二阶动量，shape = (2,)。
m_hat = m / (1.0 - beta1 ** t)  # 对一阶动量做偏差修正，shape = (2,)。
v_hat = v / (1.0 - beta2 ** t)  # 对二阶动量做偏差修正，shape = (2,)。
adam_param = adam_param - learning_rate * m_hat / (np.sqrt(v_hat) + eps)  # 按 Adam 规则更新参数。

print(\"SGD param:\", sgd_param)  # 打印 SGD 更新后的参数。
print(\"Adam param:\", adam_param)  # 打印 Adam 更新后的参数。
