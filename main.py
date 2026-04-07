# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Simple linear regression training example without PyTorch."""  # 用一句话说明这个脚本的用途。

import numpy as np  # 导入 NumPy，用来做矩阵运算、手写线性回归和梯度下降。
import matplotlib.pyplot as plt  # 导入画图库，用来画 loss 曲线和 accuracy 曲线。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，替代已废弃的 Boston 数据集。
from sklearn.model_selection import train_test_split  # 导入数据集切分工具，用于划分训练集和测试集。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，让特征更容易训练收敛。

data = fetch_california_housing()  # 下载或读取加州房价数据集。
X = data.data  # 取出输入特征矩阵，shape = (20640, 8)。
Y = data.target  # 取出每个样本对应的房价标签，shape = (20640,)。

X_train, X_test, Y_train, Y_test = train_test_split(  # 把原始数据切成训练集和测试集。
    X,  # 传入全部特征数据，shape = (20640, 8)。
    Y,  # 传入全部目标数据，shape = (20640,)。
    test_size=0.2,  # 指定 20% 的数据作为测试集。
    random_state=42,  # 固定随机种子，保证每次切分结果一致。
)

scaler = StandardScaler()  # 创建标准化器对象，只用训练集来学习均值和方差。
X_train = scaler.fit_transform(X_train)  # 在训练集上先拟合再标准化，shape 仍然是 (16512, 8)。
X_test = scaler.transform(X_test)  # 用训练集得到的标准化规则去转换测试集，shape 仍然是 (4128, 8)。

x_data = X_train.astype(np.float32)  # 把训练特征转成 float32 类型的数组，shape = (16512, 8)。
y_data = Y_train.astype(np.float32)  # 把训练标签转成 float32 类型的数组，shape = (16512,)。
x_test_data = X_test.astype(np.float32)  # 把测试特征也转成数组，shape = (4128, 8)。
y_test_data = Y_test.astype(np.float32)  # 把测试标签转成数组，shape = (4128,)。

batch_size = 32  # 设置每个 batch 里取多少条样本来训练。
input_dim = x_data.shape[1]  # 读取输入特征维度，这里等于 8。
learning_rate = 0.01  # 设置梯度下降的学习率，控制每次参数更新的步长。
epochs_num = 1000  # 设置训练轮数，让模型反复看到训练数据。
accuracy_threshold = 0.5  # 定义“预测误差小于 0.5 就算预测正确”的阈值，这是回归任务里自定义的 accuracy。

w = np.zeros(input_dim, dtype=np.float32)  # 初始化权重向量，shape = (8,)。
b = np.float32(0.0)  # 初始化偏置项，它是一个标量。

train_loss_history = []  # 记录每一轮训练集平均 loss，后面用来画图。
test_loss_history = []  # 记录每一轮测试集 loss，后面用来画图。
test_accuracy_history = []  # 记录每一轮测试集上的自定义 accuracy，后面用来画图。

for epoch in range(epochs_num):  # 训练 1000 轮，每一轮都会遍历完所有 batch。
    indices = np.random.permutation(len(x_data))  # 生成训练样本的随机顺序，作用和 DataLoader 的 shuffle=True 类似。
    x_shuffled = x_data[indices]  # 按随机顺序重排训练特征，shape 仍然是 (16512, 8)。
    y_shuffled = y_data[indices]  # 按相同顺序重排训练标签，shape 仍然是 (16512,)。
    total_loss = 0.0  # 记录这一整轮里所有 batch 的损失总和。
    num_batches = 0  # 记录这一轮总共处理了多少个 batch，后面用来算平均 loss。

    for start in range(0, len(x_shuffled), batch_size):  # 每次从打乱后的数据里切出一个 batch。
        end = start + batch_size  # 计算当前 batch 的结束位置。
        batch_x = x_shuffled[start:end]  # 取出当前 batch 的特征，shape 约是 (32, 8)。
        batch_y = y_shuffled[start:end]  # 取出当前 batch 的标签，shape 约是 (32,)。

        pred = batch_x @ w + b  # 用线性回归公式做前向传播，输出 shape = (32,)。
        error = pred - batch_y  # 计算预测值和真实值的差，shape = (32,)。
        loss = np.mean(error ** 2)  # 计算当前 batch 的均方误差损失。

        grad_w = (2.0 / len(batch_x)) * (batch_x.T @ error)  # 手动计算损失对权重的梯度，shape = (8,)。
        grad_b = (2.0 / len(batch_x)) * np.sum(error)  # 手动计算损失对偏置的梯度，它是一个标量。

        w -= learning_rate * grad_w  # 按梯度下降公式更新权重。
        b -= learning_rate * grad_b  # 按梯度下降公式更新偏置。

        total_loss += float(loss)  # 把当前 batch 的损失累加起来，方便统计整轮表现。
        num_batches += 1  # 当前 batch 处理完成，把 batch 数量加 1。

    avg_loss = total_loss / num_batches  # 计算这一轮所有 batch 的平均损失。
    train_loss_history.append(avg_loss)  # 把训练集平均 loss 记下来，方便后面画曲线。

    pred_test = x_test_data @ w + b  # 用当前模型对整个测试集做预测，输出 shape = (4128,)。
    test_error = pred_test - y_test_data  # 计算测试集预测误差，shape = (4128,)。
    test_loss = np.mean(test_error ** 2)  # 计算当前这一轮对应的测试集 loss。
    test_accuracy = np.mean(np.abs(test_error) < accuracy_threshold)  # 统计误差小于阈值的样本比例，作为回归版 accuracy。

    test_loss_history.append(float(test_loss))  # 把测试集 loss 记下来，方便后面画曲线。
    test_accuracy_history.append(float(test_accuracy))  # 把测试集 accuracy 记下来，方便后面画曲线。

    if epoch % 100 == 0:  # 每训练 100 轮打印一次日志，方便观察收敛情况。
        print(  # 输出当前轮数、训练损失、测试损失和测试 accuracy。
            f"Epoch {epoch}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_accuracy:.4f}"
        )

pred_test = x_test_data @ w + b  # 用训练好的模型对测试集做最终预测，输出 shape = (4128,)。
test_error = pred_test - y_test_data  # 计算最终测试误差，shape = (4128,)。
test_loss = np.mean(test_error ** 2)  # 计算测试集上的均方误差。
test_accuracy = np.mean(np.abs(test_error) < accuracy_threshold)  # 统计最终测试集里误差小于阈值的样本比例。

print("\nTest Loss:", float(test_loss))  # 打印最终测试损失，查看模型泛化效果。
print("Test Accuracy:", float(test_accuracy))  # 打印最终测试 accuracy，注意这里是回归任务下自定义的 accuracy。

epochs = range(1, len(train_loss_history) + 1)  # 生成横轴 epoch 序列，长度和训练轮数一致。
plt.figure(figsize=(12, 5))  # 创建一个宽 12、高 5 的画布，让两张图并排显示。

plt.subplot(1, 2, 1)  # 选择左边第 1 张子图，专门画 loss 曲线。
plt.plot(epochs, train_loss_history, label="Train Loss")  # 在左图上画训练集 loss 曲线。
plt.plot(epochs, test_loss_history, label="Test Loss")  # 在左图上画测试集 loss 曲线。
plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
plt.ylabel("Loss")  # 设置纵轴名称为 Loss。
plt.title("Loss Curve")  # 设置左图标题。
plt.legend()  # 显示图例，区分训练 loss 和测试 loss。

plt.subplot(1, 2, 2)  # 选择右边第 2 张子图，专门画 accuracy 曲线。
plt.plot(epochs, test_accuracy_history, label="Test Accuracy", color="orange")  # 在右图上画测试集 accuracy 曲线。
plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
plt.ylabel("Accuracy")  # 设置纵轴名称为 Accuracy。
plt.title("Accuracy Curve")  # 设置右图标题。
plt.legend()  # 显示图例，说明这条线表示测试 accuracy。

plt.tight_layout()  # 自动调整子图间距，避免标题和坐标轴文字重叠。
plt.show()  # 把图窗口显示出来。
