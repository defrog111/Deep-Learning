# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Simple linear regression training example with PyTorch."""  # 用一句话说明这个脚本的用途。

import torch  # 导入 PyTorch，用来定义模型、张量和训练流程。
import matplotlib.pyplot as plt  # 导入画图库，用来画 loss 曲线和 accuracy 曲线。
from torch.utils.data import DataLoader, TensorDataset  # 导入批量加载工具，把数据拆成一个个 batch。
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

x_data = torch.tensor(X_train, dtype=torch.float32)  # 把训练特征转成 float32 类型的张量，shape = (16512, 8)。
y_data = torch.tensor(Y_train, dtype=torch.float32)  # 把训练标签转成 float32 类型的张量，shape = (16512,)。
x_test_tensor = torch.tensor(X_test, dtype=torch.float32)  # 把测试特征也转成张量，shape = (4128, 8)。
y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)  # 把测试标签转成张量，shape = (4128,)。
train_dataset = TensorDataset(x_data, y_data)  # 把训练特征和标签打包成数据集，其中单个样本是 (8,) 和 ()。
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 每次取 32 个样本训练，所以 batch_x shape 约是 (32, 8)，batch_y shape 约是 (32,)。


class Net(torch.nn.Module):  # 定义一个继承自 PyTorch 模块的线性回归模型。
    def __init__(self, n_feature, n_output):  # 初始化模型时传入输入维度和输出维度。
        super().__init__()  # 调用父类初始化，让这个类具备 Module 的能力。
        self.linear = torch.nn.Linear(n_feature, n_output)  # 创建一个全连接层，输入 shape = (batch_size, 8)，输出 shape = (batch_size, 1)。

    def forward(self, x):  # 定义前向传播逻辑，x 是输入特征。
        return self.linear(x)  # 直接把输入交给线性层，返回 shape = (batch_size, 1) 的预测结果。


net = Net(x_data.shape[1], 1)  # 按训练数据的特征数创建模型，这里输入维度是 8，输出维度是 1。
loss_func = torch.nn.MSELoss()  # 定义均方误差损失函数，回归任务里最常用。
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 使用随机梯度下降优化器更新模型参数。
accuracy_threshold = 0.5  # 定义“预测误差小于 0.5 就算预测正确”的阈值，这是回归任务里自定义的 accuracy。
train_loss_history = []  # 记录每一轮训练集平均 loss，后面用来画图。
test_loss_history = []  # 记录每一轮测试集 loss，后面用来画图。
test_accuracy_history = []  # 记录每一轮测试集上的自定义 accuracy，后面用来画图。

for epoch in range(1000):  # 训练 1000 轮，每一轮都会遍历完所有 batch。
    total_loss = 0.0  # 记录这一整轮里所有 batch 的损失总和。

    for batch_x, batch_y in train_loader:  # 从 DataLoader 里一批一批取出训练数据，batch_x shape 约是 (32, 8)，batch_y shape 约是 (32,)。
        pred = net(batch_x).squeeze(1)  # 用当前 batch 做前向传播，原输出 shape = (32, 1)，压缩后变成 (32,)。
        loss = loss_func(pred, batch_y)  # 计算当前 batch 的预测误差，此时 pred 和 batch_y 的 shape 都是 (32,)。

        optimizer.zero_grad()  # 在反向传播前先清空旧梯度，避免梯度累计。
        loss.backward()  # 根据当前 batch 的损失对模型参数求梯度。
        optimizer.step()  # 使用优化器按梯度方向更新参数。

        total_loss += loss.item()  # 把当前 batch 的损失累加起来，方便统计整轮表现。

    avg_loss = total_loss / len(train_loader)  # 计算这一轮所有 batch 的平均损失。
    train_loss_history.append(avg_loss)  # 把训练集平均 loss 记下来，方便后面画曲线。

    with torch.no_grad():  # 进入不计算梯度模式，在每轮结束后评估测试集表现。
        pred_test = net(x_test_tensor).squeeze(1)  # 用当前模型对整个测试集做预测，输出 shape = (4128,)。
        test_loss = loss_func(pred_test, y_test_tensor)  # 计算当前这一轮对应的测试集 loss。
        test_accuracy = ((pred_test - y_test_tensor).abs() < accuracy_threshold).float().mean()  # 统计误差小于阈值的样本比例，作为回归版 accuracy。

    test_loss_history.append(test_loss.item())  # 把测试集 loss 记下来，方便后面画曲线。
    test_accuracy_history.append(test_accuracy.item())  # 把测试集 accuracy 记下来，方便后面画曲线。

    if epoch % 100 == 0:  # 每训练 100 轮打印一次日志，方便观察收敛情况。
        print(  # 输出当前轮数、训练损失、测试损失和测试 accuracy。
            f"Epoch {epoch}, "
            f"Train Loss: {avg_loss:.4f}, "
            f"Test Loss: {test_loss.item():.4f}, "
            f"Test Accuracy: {test_accuracy.item():.4f}"
        )

with torch.no_grad():  # 进入不计算梯度模式，评估时更省内存也更快。
    pred_test = net(x_test_tensor).squeeze(1)  # 用训练好的模型对测试集做预测，输出从 (4128, 1) 压成 (4128,)。
    test_loss = loss_func(pred_test, y_test_tensor)  # 计算测试集上的均方误差，此时两边的 shape 都是 (4128,)。
    test_accuracy = ((pred_test - y_test_tensor).abs() < accuracy_threshold).float().mean()  # 统计最终测试集里误差小于阈值的样本比例。

print("\nTest Loss:", test_loss.item())  # 打印最终测试损失，查看模型泛化效果。
print("Test Accuracy:", test_accuracy.item())  # 打印最终测试 accuracy，注意这里是回归任务下自定义的 accuracy。

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
