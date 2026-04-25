# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: one-step RNN forward.\"\"\"  # 说明这道题是手写 RNN 单步前向传播。

import numpy as np  # 导入 NumPy，用来做矩阵乘法和 tanh 激活。

x_t = np.array([[1.0, 0.0]], dtype=np.float32)  # 定义当前时刻输入向量，shape = (1, 2)。
h_prev = np.array([[0.5, -0.5]], dtype=np.float32)  # 定义上一个时刻隐藏状态，shape = (1, 2)。
w_xh = np.array([[0.2, 0.1], [0.4, -0.3]], dtype=np.float32)  # 定义输入到隐藏层权重，shape = (2, 2)。
w_hh = np.array([[0.1, 0.2], [-0.2, 0.3]], dtype=np.float32)  # 定义隐藏到隐藏权重，shape = (2, 2)。
b_h = np.zeros((1, 2), dtype=np.float32)  # 定义隐藏层偏置，shape = (1, 2)。
pre_act = x_t @ w_xh + h_prev @ w_hh + b_h  # 计算激活前线性输出，shape = (1, 2)。
h_t = np.tanh(pre_act)  # 通过 tanh 得到当前时刻隐藏状态，shape = (1, 2)。

print(\"Pre activation shape:\", pre_act.shape)  # 打印激活前张量的 shape。
print(\"Hidden state shape:\", h_t.shape)  # 打印当前隐藏状态的 shape。
