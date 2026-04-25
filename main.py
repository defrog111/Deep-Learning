# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: two-layer NumPy MLP.\"\"\"  # 说明这是手写两层 MLP。

import numpy as np  # 导入 NumPy，用来做前向传播和反向传播。

def relu(x):  # 定义 ReLU 激活函数，输入输出 shape 相同。
    return np.maximum(0.0, x)  # 返回逐元素激活结果。

def relu_grad(x):  # 定义 ReLU 的梯度函数。
    return (x > 0).astype(np.float32)  # 返回导数掩码矩阵，shape 与 x 相同。

x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32)  # 定义输入矩阵，shape = (4, 2)。
y = np.array([[0.0], [1.0], [1.0], [0.0]], dtype=np.float32)  # 定义标签矩阵，shape = (4, 1)。
w1 = np.array([[0.1, -0.2, 0.3], [0.4, 0.2, -0.5]], dtype=np.float32)  # 定义第一层权重，shape = (2, 3)。
b1 = np.zeros((1, 3), dtype=np.float32)  # 定义第一层偏置，shape = (1, 3)。
w2 = np.array([[0.2], [-0.3], [0.5]], dtype=np.float32)  # 定义第二层权重，shape = (3, 1)。
b2 = np.zeros((1, 1), dtype=np.float32)  # 定义第二层偏置，shape = (1, 1)。
learning_rate = 0.1  # 定义学习率，是标量。

for epoch in range(200):  # 开始训练循环。
    z1 = x @ w1 + b1  # 计算第一层线性输出，shape = (4, 3)。
    a1 = relu(z1)  # 计算第一层激活输出，shape = (4, 3)。
    pred = a1 @ w2 + b2  # 计算最终输出矩阵，shape = (4, 1)。
    error = pred - y  # 计算预测误差矩阵，shape = (4, 1)。
    loss = np.mean(error ** 2)  # 计算 MSE，输出是标量。
    grad_pred = (2.0 / len(x)) * error  # 计算损失对输出的梯度，shape = (4, 1)。
    grad_w2 = a1.T @ grad_pred  # 计算第二层权重梯度，shape = (3, 1)。
    grad_b2 = np.sum(grad_pred, axis=0, keepdims=True)  # 计算第二层偏置梯度，shape = (1, 1)。
    grad_a1 = grad_pred @ w2.T  # 把梯度传回隐藏层，shape = (4, 3)。
    grad_z1 = grad_a1 * relu_grad(z1)  # 结合 ReLU 导数，得到隐藏层线性输出梯度，shape = (4, 3)。
    grad_w1 = x.T @ grad_z1  # 计算第一层权重梯度，shape = (2, 3)。
    grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)  # 计算第一层偏置梯度，shape = (1, 3)。
    w2 = w2 - learning_rate * grad_w2  # 更新第二层权重。
    b2 = b2 - learning_rate * grad_b2  # 更新第二层偏置。
    w1 = w1 - learning_rate * grad_w1  # 更新第一层权重。
    b1 = b1 - learning_rate * grad_b1  # 更新第一层偏置。
    if epoch % 50 == 0:  # 每 50 轮打印一次训练损失。
        print(f\"Epoch {epoch:03d} | MSE: {loss:.4f}\")  # 打印训练日志。

print(\"Prediction shape:\", pred.shape)  # 打印最终预测矩阵的 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
pred_labels = (pred >= 0.5).astype(np.int32)  # æŠŠè¿žç»­è¾“å‡ºæŒ‰ 0.5 é˜ˆå€¼è½¬æˆäºŒåˆ†ç±»æ ‡ç­¾ï¼Œshape = (4, 1)ã€‚
y_labels = y.astype(np.int32)  # æŠŠçœŸå®žæ ‡ç­¾è½¬æˆæ•´åž‹æ ‡ç­¾çŸ©é˜µï¼Œshape = (4, 1)ã€‚
accuracy = np.mean(pred_labels == y_labels)  # è®¡ç®— Accuracyï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
tp = np.sum((pred_labels == 1) & (y_labels == 1))  # è®¡ç®— TPï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
fp = np.sum((pred_labels == 1) & (y_labels == 0))  # è®¡ç®— FPï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
fn = np.sum((pred_labels == 0) & (y_labels == 1))  # è®¡ç®— FNï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
precision = tp / (tp + fp + 1e-7)  # è®¡ç®— Precisionï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
recall = tp / (tp + fn + 1e-7)  # è®¡ç®— Recallï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # è®¡ç®— F1ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Accuracy:", float(accuracy))  # æ‰“å° Accuracyã€‚
print("Precision:", float(precision))  # æ‰“å° Precisionã€‚
print("Recall:", float(recall))  # æ‰“å° Recallã€‚
print("F1:", float(f1))  # æ‰“å° F1ã€‚
