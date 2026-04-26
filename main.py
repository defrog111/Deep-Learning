# -*- coding: utf-8 -*-  # 声明编码格式，方便整份脚本写中文注释。
"""Extra interview problem: common regression and classification metrics."""  # 用一句话说明这个脚本的用途。

import numpy as np  # 导入 NumPy，用来构造数组和手写常见指标公式。


def regression_metrics(y_true, y_pred):  # 统一计算回归常见指标，输入两个一维数组。
    error = y_true - y_pred  # 逐样本误差向量，shape = (num_samples,)。
    mse = np.mean(error ** 2)  # 计算均方误差，输出是标量，没有数组 shape。
    rmse = np.sqrt(mse)  # 对 MSE 开平方得到 RMSE，输出是标量，没有数组 shape。
    mae = np.mean(np.abs(error))  # 计算平均绝对误差，输出是标量，没有数组 shape。
    ss_res = np.sum(error ** 2)  # 计算残差平方和，输出是标量，没有数组 shape。
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 计算总平方和，输出是标量，没有数组 shape。
    r2 = 1.0 - ss_res / ss_tot  # 根据 R2 公式计算拟合优度，输出是标量，没有数组 shape。
    return {  # 返回一个字典，里面每个值都是 Python 浮点数。
        "mse": float(mse),  # 记录均方误差。
        "rmse": float(rmse),  # 记录均方根误差。
        "mae": float(mae),  # 记录平均绝对误差。
        "r2": float(r2),  # 记录 R2。
    }


def classification_metrics(y_true, y_pred):  # 统一计算二分类常见指标，输入两个一维整数数组。
    correct_mask = y_true == y_pred  # 判断每个位置是否预测正确，shape = (num_samples,)。
    accuracy = np.mean(correct_mask)  # 计算准确率，输出是标量，没有数组 shape。
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 计算真正例个数，输出是标量，没有数组 shape。
    fp = np.sum((y_true == 0) & (y_pred == 1))  # 计算假正例个数，输出是标量，没有数组 shape。
    fn = np.sum((y_true == 1) & (y_pred == 0))  # 计算假负例个数，输出是标量，没有数组 shape。
    precision = tp / (tp + fp + 1e-7)  # 计算精确率，输出是标量，没有数组 shape。
    recall = tp / (tp + fn + 1e-7)  # 计算召回率，输出是标量，没有数组 shape。
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # 计算 F1，输出是标量，没有数组 shape。
    return {  # 返回一个字典，里面每个值都是 Python 浮点数。
        "accuracy": float(accuracy),  # 记录准确率。
        "precision": float(precision),  # 记录精确率。
        "recall": float(recall),  # 记录召回率。
        "f1": float(f1),  # 记录 F1。
        "tp": int(tp),  # 记录真正例个数。
        "fp": int(fp),  # 记录假正例个数。
        "fn": int(fn),  # 记录假负例个数。
    }


y_true_reg = np.array([3.0, 4.0, 5.0], dtype=np.float32)  # 定义回归真实值向量，shape = (3,)。
y_pred_reg = np.array([2.5, 4.5, 5.5], dtype=np.float32)  # 定义回归预测值向量，shape = (3,)。
reg_metrics = regression_metrics(y_true_reg, y_pred_reg)  # 计算回归指标，返回字典，没有数组 shape。

y_true_cls = np.array([1, 0, 1, 1], dtype=np.int32)  # 定义分类真实标签向量，shape = (4,)。
y_pred_cls = np.array([1, 0, 0, 1], dtype=np.int32)  # 定义分类预测标签向量，shape = (4,)。
cls_metrics = classification_metrics(y_true_cls, y_pred_cls)  # 计算分类指标，返回字典，没有数组 shape。

print("Regression example shapes:")  # 打印回归例子的 shape 标题，这是字符串，没有 shape。
print("y_true_reg shape:", y_true_reg.shape)  # 打印回归真实值 shape，当前是 (3,)。
print("y_pred_reg shape:", y_pred_reg.shape)  # 打印回归预测值 shape，当前是 (3,)。
print("MSE:", reg_metrics["mse"])  # 打印均方误差，这是字符串和浮点数组合，没有 shape。
print("RMSE:", reg_metrics["rmse"])  # 打印均方根误差，这是字符串和浮点数组合，没有 shape。
print("MAE:", reg_metrics["mae"])  # 打印平均绝对误差，这是字符串和浮点数组合，没有 shape。
print("R2:", reg_metrics["r2"])  # 打印 R2，这是字符串和浮点数组合，没有 shape。

print("\nClassification example shapes:")  # 打印分类例子的 shape 标题，这是字符串，没有 shape。
print("y_true_cls shape:", y_true_cls.shape)  # 打印分类真实值 shape，当前是 (4,)。
print("y_pred_cls shape:", y_pred_cls.shape)  # 打印分类预测值 shape，当前是 (4,)。
print("TP:", cls_metrics["tp"])  # 打印真正例个数，这是字符串和整数组合，没有 shape。
print("FP:", cls_metrics["fp"])  # 打印假正例个数，这是字符串和整数组合，没有 shape。
print("FN:", cls_metrics["fn"])  # 打印假负例个数，这是字符串和整数组合，没有 shape。
print("Accuracy:", cls_metrics["accuracy"])  # 打印准确率，这是字符串和浮点数组合，没有 shape。
print("Precision:", cls_metrics["precision"])  # 打印精确率，这是字符串和浮点数组合，没有 shape。
print("Recall:", cls_metrics["recall"])  # 打印召回率，这是字符串和浮点数组合，没有 shape。
print("F1:", cls_metrics["f1"])  # 打印 F1，这是字符串和浮点数组合，没有 shape。
