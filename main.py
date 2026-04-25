# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: NumPy self-attention.\"\"\"  # 说明这道题是手写 self-attention。

import numpy as np  # 导入 NumPy，用来做矩阵乘法和 softmax。

def softmax(x):  # 定义 softmax 函数，通常沿最后一维做归一化。
    shifted = x - np.max(x, axis=-1, keepdims=True)  # 先减去每行最大值，避免数值溢出。
    exp_x = np.exp(shifted)  # 对平移后的矩阵逐元素取指数，shape 与输入相同。
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)  # 返回概率矩阵，shape 与输入相同。

x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)  # 定义输入序列矩阵，shape = (3, 2)。
w_q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # 定义 Query 投影矩阵，shape = (2, 2)。
w_k = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32)  # 定义 Key 投影矩阵，shape = (2, 2)。
w_v = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # 定义 Value 投影矩阵，shape = (2, 2)。
q = x @ w_q  # 计算 Query 矩阵 q，shape = (3, 2)。
k = x @ w_k  # 计算 Key 矩阵 k，shape = (3, 2)。
v = x @ w_v  # 计算 Value 矩阵 v，shape = (3, 2)。
scores = (q @ k.T) / np.sqrt(k.shape[1])  # 计算缩放点积注意力分数矩阵，shape = (3, 3)。
weights = softmax(scores)  # 对分数矩阵做 softmax，得到权重矩阵，shape = (3, 3)。
output = weights @ v  # 用权重矩阵加权 Value，得到最终输出，shape = (3, 2)。

print(\"Score shape:\", scores.shape)  # 打印分数矩阵 shape。
print(\"Weight shape:\", weights.shape)  # 打印权重矩阵 shape。
print(\"Output shape:\", output.shape)  # 打印输出矩阵 shape。
