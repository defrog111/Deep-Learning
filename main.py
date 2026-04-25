# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: softmax plus cross entropy.\"\"\"  # 说明这道题是手写 softmax 和交叉熵。

import numpy as np  # 导入 NumPy，用来做概率归一化和损失计算。

def softmax(logits):  # 定义 softmax 函数，输入 logits 的 shape = (num_samples, num_classes)。
    shifted = logits - np.max(logits, axis=1, keepdims=True)  # 每行减最大值，避免指数溢出。
    exp_logits = np.exp(shifted)  # 对平移后的 logits 逐元素取指数，shape 不变。
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # 返回概率矩阵，shape 不变。

logits = np.array([[2.0, 1.0, 0.1], [0.5, 1.5, 2.5]], dtype=np.float32)  # 定义两条样本的分类分数矩阵，shape = (2, 3)。
labels = np.array([0, 2], dtype=np.int32)  # 定义真实类别编号向量，shape = (2,)。
probs = softmax(logits)  # 计算每个类别的概率矩阵，shape = (2, 3)。
correct_probs = probs[np.arange(len(labels)), labels]  # 取出真实类别对应的概率，shape = (2,)。
loss = -np.mean(np.log(correct_probs + 1e-7))  # 计算平均交叉熵损失，输出是标量。
pred_labels = np.argmax(probs, axis=1)  # 取每行概率最大的类别编号，shape = (2,)。

print(\"Probability shape:\", probs.shape)  # 打印概率矩阵 shape。
print(\"Predicted labels:\", pred_labels)  # 打印预测类别。
print(\"Cross entropy loss:\", float(loss))  # 打印交叉熵损失。
