# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便整份脚本写中文注释。
"""NumPy linear regression example with train/val/test split."""  # 用一句话说明这个脚本的用途。

import numpy as np  # 导入 NumPy，用来做数组运算、矩阵乘法和手写梯度下降。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，作为表格回归示例。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分 train、val、test。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，只能在训练集上 fit。


def prepare_data(random_state=42):  # 负责读数据、切分数据、标准化数据和类型转换。
    dataset = fetch_california_housing()  # 读取加州房价数据集对象，这里不是张量，没有 shape。
    x = dataset.data  # 取出特征矩阵，shape = (20640, 8)。
    y = dataset.target  # 取出回归标签向量，shape = (20640,)。

    x_train_full, x_test, y_train_full, y_test = train_test_split(  # 先切出训练+验证集合和测试集合。
        x,  # 传入全部特征矩阵，shape = (20640, 8)。
        y,  # 传入全部标签向量，shape = (20640,)。
        test_size=0.2,  # 取 20% 的样本作为测试集，这不是张量，没有 shape。
        random_state=random_state,  # 固定随机种子，保证切分结果稳定，这不是张量，没有 shape。
    )  # 切分后 x_train_full shape = (16512, 8)，x_test shape = (4128, 8)。

    x_train, x_val, y_train, y_val = train_test_split(  # 再从训练+验证集合中切出真正的训练集和验证集。
        x_train_full,  # 传入训练+验证特征矩阵，shape = (16512, 8)。
        y_train_full,  # 传入训练+验证标签向量，shape = (16512,)。
        test_size=0.2,  # 继续取 20% 作为验证集，这不是张量，没有 shape。
        random_state=random_state,  # 固定随机种子，保证切分结果稳定，这不是张量，没有 shape。
    )  # 切分后 x_train shape 约是 (13209, 8)，x_val shape 约是 (3303, 8)。

    scaler = StandardScaler()  # 创建标准化器对象，这里不是张量，没有 shape。
    x_train = scaler.fit_transform(x_train)  # 在训练集上学习均值和方差并标准化，shape 仍然是 (13209, 8) 左右。
    x_val = scaler.transform(x_val)  # 用训练集统计量标准化验证集，shape 仍然是 (3303, 8) 左右。
    x_test = scaler.transform(x_test)  # 用训练集统计量标准化测试集，shape 仍然是 (4128, 8) 左右。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32 数组，shape = (13209, 8) 左右。
    x_val = x_val.astype(np.float32)  # 把验证特征转成 float32 数组，shape = (3303, 8) 左右。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32 数组，shape = (4128, 8) 左右。
    y_train = y_train.astype(np.float32)  # 把训练标签转成 float32 数组，shape = (13209,) 左右。
    y_val = y_val.astype(np.float32)  # 把验证标签转成 float32 数组，shape = (3303,) 左右。
    y_test = y_test.astype(np.float32)  # 把测试标签转成 float32 数组，shape = (4128,) 左右。
    return x_train, x_val, x_test, y_train, y_val, y_test  # 返回 train、val、test 三份数据。


def regression_metrics(pred, target):  # 统一计算回归指标，避免只看一个 loss。
    error = pred - target  # 逐样本误差向量，shape = (num_samples,)。
    mse = np.mean(error ** 2)  # 计算均方误差，输出是标量，没有数组 shape。
    rmse = np.sqrt(mse)  # 对 MSE 开平方得到 RMSE，输出是标量，没有数组 shape。
    mae = np.mean(np.abs(error))  # 计算平均绝对误差，输出是标量，没有数组 shape。
    ss_res = np.sum(error ** 2)  # 计算残差平方和，输出是标量，没有数组 shape。
    ss_tot = np.sum((target - np.mean(target)) ** 2)  # 计算总平方和，输出是标量，没有数组 shape。
    r2 = 1.0 - ss_res / ss_tot  # 根据 R2 公式计算拟合优度，输出是标量，没有数组 shape。
    return {  # 返回一个字典，里面每个值都是 Python 浮点数。
        "mse": float(mse),  # 记录均方误差。
        "rmse": float(rmse),  # 记录均方根误差。
        "mae": float(mae),  # 记录平均绝对误差。
        "r2": float(r2),  # 记录 R2。
    }


def train_linear_regression(x_train, y_train, x_val, y_val, learning_rate=0.01, epochs=300):  # 手写 full-batch 线性回归训练函数。
    w = np.zeros(x_train.shape[1], dtype=np.float32)  # 初始化权重向量，shape = (8,)。
    b = np.float32(0.0)  # 初始化偏置标量，这里不是数组，没有 shape。
    n = len(x_train)  # 训练样本数，是整数，没有 shape。
    train_loss_history = []  # 保存每一轮训练集 MSE，列表长度最后会等于 epochs。
    val_loss_history = []  # 保存每一轮验证集 MSE，列表长度最后会等于 epochs。
    train_r2_history = []  # 保存每一轮训练集 R2，列表长度最后会等于 epochs。
    val_r2_history = []  # 保存每一轮验证集 R2，列表长度最后会等于 epochs。

    for epoch in range(epochs):  # 外层循环控制训练轮数，epoch 是整数，没有 shape。
        train_pred = x_train @ w + b  # 前向传播得到训练集预测值，shape = (num_train,)。
        train_error = train_pred - y_train  # 训练集误差向量，shape = (num_train,)。
        train_loss = np.mean(train_error ** 2)  # 训练集 MSE，输出是标量，没有数组 shape。

        grad_w = (2.0 / n) * (x_train.T @ train_error)  # 计算损失对权重的梯度，shape = (8,)。
        grad_b = (2.0 / n) * np.sum(train_error)  # 计算损失对偏置的梯度，输出是标量，没有数组 shape。

        w -= learning_rate * grad_w  # 用梯度下降更新权重向量，shape 仍然是 (8,)。
        b -= learning_rate * grad_b  # 用梯度下降更新偏置标量，这里不是数组，没有 shape。

        train_metrics = regression_metrics(train_pred, y_train)  # 计算当前轮训练集指标，返回字典，没有数组 shape。
        val_pred = x_val @ w + b  # 用更新后的参数对验证集做预测，shape = (num_val,)。
        val_metrics = regression_metrics(val_pred, y_val)  # 计算当前轮验证集指标，返回字典，没有数组 shape。

        train_loss_history.append(train_metrics["mse"])  # 记录当前轮训练集 MSE，列表长度加 1。
        val_loss_history.append(val_metrics["mse"])  # 记录当前轮验证集 MSE，列表长度加 1。
        train_r2_history.append(train_metrics["r2"])  # 记录当前轮训练集 R2，列表长度加 1。
        val_r2_history.append(val_metrics["r2"])  # 记录当前轮验证集 R2，列表长度加 1。

        if epoch % 50 == 0 or epoch == epochs - 1:  # 每 50 轮和最后一轮打印一次日志，这里不是张量，没有 shape。
            print(  # 打印当前轮训练日志，这里不是张量，没有 shape。
                f"Epoch {epoch:03d} | "  # 打印轮数信息，这是字符串，没有 shape。
                f"Train Loss: {train_metrics['mse']:.4f} | "  # 打印训练集 MSE，这是字符串，没有 shape。
                f"Val Loss: {val_metrics['mse']:.4f} | "  # 打印验证集 MSE，这是字符串，没有 shape。
                f"Train R2: {train_metrics['r2']:.4f} | "  # 打印训练集 R2，这是字符串，没有 shape。
                f"Val R2: {val_metrics['r2']:.4f}"  # 打印验证集 R2，这是字符串，没有 shape。
            )

    history = {  # 把历史曲线数据收进字典，便于后面统一打印或扩展。
        "train_loss": train_loss_history,  # 训练集 loss 历史，列表长度等于 epochs。
        "val_loss": val_loss_history,  # 验证集 loss 历史，列表长度等于 epochs。
        "train_r2": train_r2_history,  # 训练集 R2 历史，列表长度等于 epochs。
        "val_r2": val_r2_history,  # 验证集 R2 历史，列表长度等于 epochs。
    }
    return w, b, history  # 返回训练好的权重、偏置和历史记录。


def predict(x_data, w, b):  # 单独封装预测逻辑，让主流程更清楚。
    return x_data @ w + b  # 矩阵乘法后再加偏置，输出 shape = (num_samples,)。


def main():  # 主入口函数，把完整流程串起来。
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data()  # 准备 train、val、test 三份数据。
    w, b, history = train_linear_regression(x_train, y_train, x_val, y_val)  # 训练手写线性回归模型。

    train_pred = predict(x_train, w, b)  # 对训练集做最终预测，shape = (num_train,)。
    val_pred = predict(x_val, w, b)  # 对验证集做最终预测，shape = (num_val,)。
    test_pred = predict(x_test, w, b)  # 对测试集做最终预测，shape = (num_test,)。

    train_metrics = regression_metrics(train_pred, y_train)  # 计算训练集最终指标，返回字典，没有数组 shape。
    val_metrics = regression_metrics(val_pred, y_val)  # 计算验证集最终指标，返回字典，没有数组 shape。
    test_metrics = regression_metrics(test_pred, y_test)  # 计算测试集最终指标，返回字典，没有数组 shape。

    print(f"\nWeight shape: {w.shape}")  # 打印权重向量 shape，当前一般是 (8,)。
    print("Bias shape: scalar")  # 打印偏置是标量，不是数组，没有 shape。
    print(f"Train prediction shape: {train_pred.shape}")  # 打印训练集预测 shape，当前一般是 (13209,) 左右。
    print(f"Val prediction shape: {val_pred.shape}")  # 打印验证集预测 shape，当前一般是 (3303,) 左右。
    print(f"Test prediction shape: {test_pred.shape}")  # 打印测试集预测 shape，当前一般是 (4128,) 左右。

    print("\nFinal train metrics:")  # 打印训练集指标标题，这是字符串，没有 shape。
    print("Train MSE:", train_metrics["mse"])  # 打印训练集 MSE，这是字符串和浮点数组合，没有 shape。
    print("Train RMSE:", train_metrics["rmse"])  # 打印训练集 RMSE，这是字符串和浮点数组合，没有 shape。
    print("Train MAE:", train_metrics["mae"])  # 打印训练集 MAE，这是字符串和浮点数组合，没有 shape。
    print("Train R2:", train_metrics["r2"])  # 打印训练集 R2，这是字符串和浮点数组合，没有 shape。

    print("\nFinal val metrics:")  # 打印验证集指标标题，这是字符串，没有 shape。
    print("Val MSE:", val_metrics["mse"])  # 打印验证集 MSE，这是字符串和浮点数组合，没有 shape。
    print("Val RMSE:", val_metrics["rmse"])  # 打印验证集 RMSE，这是字符串和浮点数组合，没有 shape。
    print("Val MAE:", val_metrics["mae"])  # 打印验证集 MAE，这是字符串和浮点数组合，没有 shape。
    print("Val R2:", val_metrics["r2"])  # 打印验证集 R2，这是字符串和浮点数组合，没有 shape。

    print("\nFinal test metrics:")  # 打印测试集指标标题，这是字符串，没有 shape。
    print("Test MSE:", test_metrics["mse"])  # 打印测试集 MSE，这是字符串和浮点数组合，没有 shape。
    print("Test RMSE:", test_metrics["rmse"])  # 打印测试集 RMSE，这是字符串和浮点数组合，没有 shape。
    print("Test MAE:", test_metrics["mae"])  # 打印测试集 MAE，这是字符串和浮点数组合，没有 shape。
    print("Test R2:", test_metrics["r2"])  # 打印测试集 R2，这是字符串和浮点数组合，没有 shape。

    print("\nHistory length:")  # 打印历史记录标题，这是字符串，没有 shape。
    print("Train loss history length:", len(history["train_loss"]))  # 打印训练集 loss 历史长度，这是整数，没有 shape。
    print("Val loss history length:", len(history["val_loss"]))  # 打印验证集 loss 历史长度，这是整数，没有 shape。
    print("Train R2 history length:", len(history["train_r2"]))  # 打印训练集 R2 历史长度，这是整数，没有 shape。
    print("Val R2 history length:", len(history["val_r2"]))  # 打印验证集 R2 历史长度，这是整数，没有 shape。


if __name__ == "__main__":  # 只有直接运行这个文件时才执行 main，被导入时不会自动运行。
    main()  # 调用主函数，开始完整的手写线性回归流程，这里不是张量，没有 shape。
