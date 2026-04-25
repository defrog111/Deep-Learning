# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: common metrics.\"\"\"  # 说明这道题是手写常见指标。

import numpy as np  # 导入 NumPy，用来做指标计算。

y_true_reg = np.array([3.0, 4.0, 5.0], dtype=np.float32)  # 定义回归真实值向量，shape = (3,)。
y_pred_reg = np.array([2.5, 4.5, 5.5], dtype=np.float32)  # 定义回归预测值向量，shape = (3,)。
mse = np.mean((y_true_reg - y_pred_reg) ** 2)  # 计算 MSE，输出是标量。
mae = np.mean(np.abs(y_true_reg - y_pred_reg))  # 计算 MAE，输出是标量。
ss_res = np.sum((y_true_reg - y_pred_reg) ** 2)  # 计算残差平方和，输出是标量。
ss_tot = np.sum((y_true_reg - np.mean(y_true_reg)) ** 2)  # 计算总平方和，输出是标量。
r2 = 1.0 - ss_res / ss_tot  # 计算回归 R2，输出是标量。

y_true_cls = np.array([1, 0, 1, 1], dtype=np.int32)  # 定义分类真实标签向量，shape = (4,)。
y_pred_cls = np.array([1, 0, 0, 1], dtype=np.int32)  # 定义分类预测标签向量，shape = (4,)。
accuracy = np.mean(y_true_cls == y_pred_cls)  # 计算 Accuracy，输出是标量。
tp = np.sum((y_true_cls == 1) & (y_pred_cls == 1))  # 计算 TP，输出是标量。
fp = np.sum((y_true_cls == 0) & (y_pred_cls == 1))  # 计算 FP，输出是标量。
fn = np.sum((y_true_cls == 1) & (y_pred_cls == 0))  # 计算 FN，输出是标量。
precision = tp / (tp + fp + 1e-7)  # 计算 Precision，输出是标量。
recall = tp / (tp + fn + 1e-7)  # 计算 Recall，输出是标量。
f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # 计算 F1，输出是标量。

print(\"MSE:\", float(mse))  # 打印 MSE。
print(\"MAE:\", float(mae))  # 打印 MAE。
print(\"R2:\", float(r2))  # 打印 R2。
print(\"Accuracy:\", float(accuracy))  # 打印 Accuracy。
print(\"Precision:\", float(precision))  # 打印 Precision。
print(\"Recall:\", float(recall))  # 打印 Recall。
print(\"F1:\", float(f1))  # 打印 F1。
