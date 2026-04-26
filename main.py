# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便整份脚本写中文注释。
"""Main branch PyTorch linear regression example with train/val/test split."""  # 用一句话说明这个脚本的用途。

import matplotlib.pyplot as plt  # 导入 Matplotlib，用来画 loss 曲线和 R2 曲线。
import torch  # 导入 PyTorch，用来定义张量、模型、损失函数和优化器。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，作为表格回归示例。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分 train、val、test。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，只能在训练集上 fit。
from torch.utils.data import DataLoader, TensorDataset  # 导入批量加载工具，把训练集拆成一个个 batch。


class Net(torch.nn.Module):  # 定义一个最简单的线性回归网络，输入 shape 常见是 (batch_size, 8)。
    def __init__(self, input_dim, output_dim=1):  # 初始化网络时传入输入维度和输出维度，这里都不是张量，没有 shape。
        super().__init__()  # 调用父类初始化函数，让当前类具备 Module 能力，这里不是张量，没有 shape。
        self.linear = torch.nn.Linear(input_dim, output_dim)  # 创建全连接层，权重 shape = (1, 8)，偏置 shape = (1,)。

    def forward(self, x):  # 定义前向传播函数，输入 x 的 shape 常见是 (batch_size, 8) 或 (N, 8)。
        return self.linear(x)  # 返回线性层输出，shape = (batch_size, 1) 或 (N, 1)。


def regression_metrics(pred, target):  # 统一计算回归指标，避免只看一个 loss。
    error = pred - target  # 逐样本误差向量，shape = (num_samples,)。
    mse = torch.mean(error ** 2)  # 计算均方误差，输出是 0 维标量张量，shape = ()。
    mae = torch.mean(torch.abs(error))  # 计算平均绝对误差，输出是 0 维标量张量，shape = ()。
    rmse = torch.sqrt(mse)  # 对 MSE 开平方得到 RMSE，输出是 0 维标量张量，shape = ()。
    ss_res = torch.sum(error ** 2)  # 计算残差平方和，输出是 0 维标量张量，shape = ()。
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)  # 计算总平方和，输出是 0 维标量张量，shape = ()。
    r2 = 1 - ss_res / ss_tot  # 根据 R2 公式计算拟合优度，输出是 0 维标量张量，shape = ()。
    return {  # 返回一个字典，里面每个值都是 Python 浮点数。
        "mse": mse.item(),  # 记录均方误差。
        "mae": mae.item(),  # 记录平均绝对误差。
        "rmse": rmse.item(),  # 记录均方根误差。
        "r2": r2.item(),  # 记录 R2。
    }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择运行设备，这里不是张量，没有 shape。
print(f"Using device: {device}")  # 打印当前设备，这是字符串，没有 shape。

dataset = fetch_california_housing()  # 读取加州房价数据集对象，这里不是张量，没有 shape。
features = dataset.data  # 取出特征矩阵，shape = (20640, 8)。
targets = dataset.target  # 取出标签向量，shape = (20640,)。

x_train_full, x_test, y_train_full, y_test = train_test_split(  # 先切出训练+验证集合和测试集合。
    features,  # 传入全部特征矩阵，shape = (20640, 8)。
    targets,  # 传入全部标签向量，shape = (20640,)。
    test_size=0.2,  # 取 20% 作为测试集，这里不是张量，没有 shape。
    random_state=42,  # 固定随机种子，保证切分结果稳定，这里不是张量，没有 shape。
)  # 切分后 x_train_full shape = (16512, 8)，x_test shape = (4128, 8)。
x_train, x_val, y_train, y_val = train_test_split(  # 再从训练+验证集合中切出训练集和验证集。
    x_train_full,  # 传入训练+验证特征矩阵，shape = (16512, 8)。
    y_train_full,  # 传入训练+验证标签向量，shape = (16512,)。
    test_size=0.2,  # 继续取 20% 作为验证集，这里不是张量，没有 shape。
    random_state=42,  # 固定随机种子，保证切分结果稳定，这里不是张量，没有 shape。
)  # 切分后 x_train shape 约是 (13209, 8)，x_val shape 约是 (3303, 8)。

scaler = StandardScaler()  # 创建标准化器对象，这里不是张量，没有 shape。
x_train = scaler.fit_transform(x_train)  # 在训练集上拟合并标准化，shape 仍然是 (13209, 8) 左右。
x_val = scaler.transform(x_val)  # 用训练集统计量标准化验证集，shape 仍然是 (3303, 8) 左右。
x_test = scaler.transform(x_test)  # 用训练集统计量标准化测试集，shape 仍然是 (4128, 8) 左右。

x_train = torch.tensor(x_train, dtype=torch.float32)  # 把训练特征转成张量，shape = (13209, 8) 左右。
y_train = torch.tensor(y_train, dtype=torch.float32)  # 把训练标签转成张量，shape = (13209,) 左右。
x_val = torch.tensor(x_val, dtype=torch.float32).to(device)  # 把验证特征转成张量并搬到 device，shape = (3303, 8) 左右。
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)  # 把验证标签转成张量并搬到 device，shape = (3303,) 左右。
x_test = torch.tensor(x_test, dtype=torch.float32).to(device)  # 把测试特征转成张量并搬到 device，shape = (4128, 8) 左右。
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)  # 把测试标签转成张量并搬到 device，shape = (4128,) 左右。

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)  # 创建训练加载器，batch_x shape 常见是 (32, 8)，batch_y shape 常见是 (32,)。
x_train_eval = x_train.to(device)  # 复制一份训练特征到 device 上做整集评估，shape = (13209, 8) 左右。
y_train_eval = y_train.to(device)  # 复制一份训练标签到 device 上做整集评估，shape = (13209,) 左右。

net = Net(x_train.shape[1], 1).to(device)  # 创建网络并搬到 device，net(batch_x) 输出 shape 常见是 (batch_size, 1)。
loss_func = torch.nn.MSELoss()  # 定义均方误差损失函数，这里不是张量，没有 shape。
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 定义 SGD 优化器，这里不是张量，没有 shape。
train_loss_history = []  # 保存每一轮训练集 MSE，列表长度最后会等于 epoch 数。
val_loss_history = []  # 保存每一轮验证集 MSE，列表长度最后会等于 epoch 数。
train_r2_history = []  # 保存每一轮训练集 R2，列表长度最后会等于 epoch 数。
val_r2_history = []  # 保存每一轮验证集 R2，列表长度最后会等于 epoch 数。

for epoch in range(300):  # 训练 300 轮，epoch 是整数，没有 shape。
    net.train()  # 切换到训练模式，这里不是张量，没有 shape。

    for batch_x, batch_y in train_loader:  # 从 DataLoader 中逐批取数据，batch_x shape 常见是 (32, 8)，batch_y shape 常见是 (32,)。
        batch_x = batch_x.to(device)  # 把当前批次特征搬到 device，上下文 shape 仍然是 (32, 8)。
        batch_y = batch_y.to(device)  # 把当前批次标签搬到 device，上下文 shape 仍然是 (32,)。
        pred = net(batch_x).squeeze(1)  # 先前向传播得到 (32, 1)，再压掉最后一维得到 (32,)。
        loss = loss_func(pred, batch_y)  # 计算当前 batch 的 MSE，输出是 0 维标量张量，shape = ()。

        optimizer.zero_grad()  # 清空上一轮留下的梯度，这里不是张量，没有 shape。
        loss.backward()  # 对当前 batch 的 loss 做反向传播，这里不是显式新张量，没有 shape。
        optimizer.step()  # 按梯度更新参数，这里不是张量，没有 shape。

    net.eval()  # 切换到评估模式，这里不是张量，没有 shape。
    with torch.no_grad():  # 关闭梯度计算，这里不是张量，没有 shape。
        train_pred = net(x_train_eval).squeeze(1)  # 对训练集整集做预测，shape = (13209,) 左右。
        val_pred = net(x_val).squeeze(1)  # 对验证集整集做预测，shape = (3303,) 左右。
        train_metrics = regression_metrics(train_pred, y_train_eval)  # 计算训练集指标，返回字典，没有 shape。
        val_metrics = regression_metrics(val_pred, y_val)  # 计算验证集指标，返回字典，没有 shape。

    train_loss_history.append(train_metrics["mse"])  # 记录训练集 MSE，列表长度加 1。
    val_loss_history.append(val_metrics["mse"])  # 记录验证集 MSE，列表长度加 1。
    train_r2_history.append(train_metrics["r2"])  # 记录训练集 R2，列表长度加 1。
    val_r2_history.append(val_metrics["r2"])  # 记录验证集 R2，列表长度加 1。

    if epoch % 50 == 0 or epoch == 299:  # 每 50 轮和最后一轮打印一次日志，这里不是张量，没有 shape。
        print(  # 打印当前轮训练日志，这里不是张量，没有 shape。
            f"Epoch {epoch:03d} | "  # 打印轮数，这是字符串，没有 shape。
            f"Train Loss: {train_metrics['mse']:.4f} | "  # 打印训练集 MSE，这是字符串，没有 shape。
            f"Val Loss: {val_metrics['mse']:.4f} | "  # 打印验证集 MSE，这是字符串，没有 shape。
            f"Train R2: {train_metrics['r2']:.4f} | "  # 打印训练集 R2，这是字符串，没有 shape。
            f"Val R2: {val_metrics['r2']:.4f}"  # 打印验证集 R2，这是字符串，没有 shape。
        )

with torch.no_grad():  # 关闭梯度计算，准备做最终测试，这里不是张量，没有 shape。
    test_pred = net(x_test).squeeze(1)  # 对测试集整集做预测，shape = (4128,) 左右。
    test_metrics = regression_metrics(test_pred, y_test)  # 计算测试集指标，返回字典，没有 shape。

print("\nFinal test metrics:")  # 打印测试集指标标题，这是字符串，没有 shape。
print("Test MSE:", test_metrics["mse"])  # 打印测试集 MSE，这是字符串和浮点数组合，没有 shape。
print("Test MAE:", test_metrics["mae"])  # 打印测试集 MAE，这是字符串和浮点数组合，没有 shape。
print("Test RMSE:", test_metrics["rmse"])  # 打印测试集 RMSE，这是字符串和浮点数组合，没有 shape。
print("Test R2:", test_metrics["r2"])  # 打印测试集 R2，这是字符串和浮点数组合，没有 shape。

epochs = range(1, len(train_loss_history) + 1)  # 创建横轴 epoch 序列，长度等于 300，这里不是张量，没有 shape。
plt.figure(figsize=(12, 5))  # 创建宽 12、高 5 的画布，这里不是张量，没有 shape。

plt.subplot(1, 2, 1)  # 选择左边第 1 张子图，这里不是张量，没有 shape。
plt.plot(epochs, train_loss_history, label="Train Loss")  # 画训练集 loss 曲线，横轴长度和纵轴长度都等于 300。
plt.plot(epochs, val_loss_history, label="Val Loss")  # 画验证集 loss 曲线，横轴长度和纵轴长度都等于 300。
plt.xlabel("Epoch")  # 设置横轴名称，这里不是张量，没有 shape。
plt.ylabel("MSE")  # 设置纵轴名称，这里不是张量，没有 shape。
plt.title("Loss Curve")  # 设置标题，这里不是张量，没有 shape。
plt.legend()  # 显示图例，这里不是张量，没有 shape。

plt.subplot(1, 2, 2)  # 选择右边第 2 张子图，这里不是张量，没有 shape。
plt.plot(epochs, train_r2_history, label="Train R2")  # 画训练集 R2 曲线，横轴长度和纵轴长度都等于 300。
plt.plot(epochs, val_r2_history, label="Val R2")  # 画验证集 R2 曲线，横轴长度和纵轴长度都等于 300。
plt.xlabel("Epoch")  # 设置横轴名称，这里不是张量，没有 shape。
plt.ylabel("R2")  # 设置纵轴名称，这里不是张量，没有 shape。
plt.title("R2 Curve")  # 设置标题，这里不是张量，没有 shape。
plt.legend()  # 显示图例，这里不是张量，没有 shape。

plt.tight_layout()  # 自动调整子图间距，这里不是张量，没有 shape。
plt.show()  # 显示图像窗口，这里不是张量，没有 shape。
