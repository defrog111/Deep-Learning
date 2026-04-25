# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: NumPy logistic regression.\"\"\"  # 说明这是手写逻辑回归题目。

import numpy as np  # 导入 NumPy，用来做向量计算和 sigmoid；这里没有张量 shape。

def sigmoid(x):  # 定义 sigmoid 函数，输入输出 shape 相同。
    return 1.0 / (1.0 + np.exp(-x))  # 返回归一化到 0 到 1 之间的概率。

x_train = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=np.float32)  # 定义训练特征矩阵，shape = (4, 2)。
y_train = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)  # 定义训练标签向量，shape = (4,)。

w = np.zeros(x_train.shape[1], dtype=np.float32)  # 初始化权重向量，shape = (2,)。
b = np.float32(0.0)  # 初始化偏置标量。
learning_rate = 0.1  # 定义学习率，是标量。
epochs = 300  # 定义训练轮数，是整数。
n = x_train.shape[0]  # 记录训练样本数 n = 4。

eps = 1e-7  # 定义很小的常数，避免出现 log(0)。
for epoch in range(epochs):  # 开始训练循环。
    logits = x_train @ w + b  # 计算线性输出 logits，shape = (4,)。
    probs = sigmoid(logits)  # 计算预测概率 probs，shape = (4,)。
    loss = -np.mean(y_train * np.log(probs + eps) + (1.0 - y_train) * np.log(1.0 - probs + eps))  # 计算二分类交叉熵，输出是标量。
    error = probs - y_train  # 计算概率误差向量，shape = (4,)。
    grad_w = (1.0 / n) * (x_train.T @ error)  # 计算权重梯度，shape = (2,)。
    grad_b = (1.0 / n) * np.sum(error)  # 计算偏置梯度，输出是标量。
    w = w - learning_rate * grad_w  # 更新权重。
    b = b - learning_rate * grad_b  # 更新偏置。
    if epoch % 60 == 0:  # 每 60 轮打印一次日志。
        print(f\"Epoch {epoch:03d} | BCE Loss: {loss:.4f}\")  # 打印训练损失。

x_test = np.array([[1.5, 1.0], [2.5, 1.0]], dtype=np.float32)  # 定义测试特征矩阵，shape = (2, 2)。
probs_test = sigmoid(x_test @ w + b)  # 计算测试概率向量，shape = (2,)。
pred_labels = (probs_test >= 0.5).astype(np.int32)  # 按 0.5 阈值转成预测类别，shape = (2,)。

print(\"Probability shape:\", probs_test.shape)  # 打印概率向量的 shape。
print(\"Predicted labels:\", pred_labels)  # 打印预测类别。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
y_true = np.array([0, 1], dtype=np.int32)  # æž„é€ ä¸€ç»„æµ‹è¯•çœŸå®žæ ‡ç­¾ï¼Œshape = (2,)ã€‚
accuracy = np.mean(pred_labels == y_true)  # è®¡ç®— Accuracyï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
tp = np.sum((pred_labels == 1) & (y_true == 1))  # è®¡ç®— TPï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
fp = np.sum((pred_labels == 1) & (y_true == 0))  # è®¡ç®— FPï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
fn = np.sum((pred_labels == 0) & (y_true == 1))  # è®¡ç®— FNï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
precision = tp / (tp + fp + 1e-7)  # è®¡ç®— Precisionï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
recall = tp / (tp + fn + 1e-7)  # è®¡ç®— Recallï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # è®¡ç®— F1ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Accuracy:", float(accuracy))  # æ‰“å° Accuracyã€‚
print("Precision:", float(precision))  # æ‰“å° Precisionã€‚
print("Recall:", float(recall))  # æ‰“å° Recallã€‚
print("F1:", float(f1))  # æ‰“å° F1ã€‚
