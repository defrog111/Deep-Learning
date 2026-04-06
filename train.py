import torch  # 导入 PyTorch 主库，用来处理张量计算。
import torch.nn as nn  # 导入神经网络模块，用来定义线性回归模型。
import torch.optim as optim  # 导入优化器模块，用来更新模型参数。


class LinearRegressionModel(nn.Module):  # 定义一个线性回归模型类。
    def __init__(self):  # 初始化模型结构。
        super().__init__()  # 调用父类初始化方法。
        self.linear = nn.Linear(2, 1)  # 定义一个输入维度为 2、输出维度为 1 的线性层。

    def forward(self, x):  # 定义前向传播逻辑。
        return self.linear(x)  # 返回线性层的预测结果。


torch.manual_seed(42)  # 固定随机种子，保证每次运行时结果更容易复现。

print("MPS available:", torch.backends.mps.is_available())  # 打印当前设备是否支持 Apple MPS 加速。
print("MPS built:", torch.backends.mps.is_built())  # 打印当前 PyTorch 是否编译了 MPS 支持。

# 生成第一列特征，表示第一个输入变量。
x1 = torch.linspace(-10, 10, 100).unsqueeze(1)
# 生成第二列特征，表示第二个输入变量。
x2 = torch.randn(100, 1) * 3
# 把两个特征按列拼接起来，得到形状为 [100, 2] 的输入数据。
X = torch.cat((x1, x2), dim=1)
# 按照 y = 2*x1 + 5*x2 + 1 的真实关系生成目标值，并加入少量随机噪声。
y = 2 * x1 + 5 * x2 + 1 + torch.randn(100, 1) * 2

# 设置训练集所占比例为 80%。
train_ratio = 0.8
# 计算训练集的样本数量。
train_size = int(len(X) * train_ratio)

# 前 80 个样本作为训练集输入。
X_train = X[:train_size]
# 前 80 个样本作为训练集标签。
y_train = y[:train_size]

print("Input shape:", X.shape)  # 打印整个输入数据的形状。
print("Train input shape:", X_train.shape)  # 打印训练集输入的形状。
print("Each sample has features:", X.shape[1])  # 打印每个样本包含多少个特征。

model = LinearRegressionModel()  # 创建线性回归模型实例。
criterion = nn.MSELoss()  # 定义均方误差损失函数。
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 定义随机梯度下降优化器。

epochs = 500  # 设置训练轮数。

for epoch in range(epochs):  # 开始训练循环。
    model.train()  # 将模型切换到训练模式。
    predictions = model(X_train)  # 使用训练集输入得到预测结果。
    loss = criterion(predictions, y_train)  # 计算训练集上的损失。

    optimizer.zero_grad()  # 清空上一轮的梯度。
    loss.backward()  # 反向传播，计算当前梯度。
    optimizer.step()  # 根据梯度更新模型参数。

    if (epoch + 1) % 50 == 0:  # 每 50 轮打印一次训练损失。
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}")  # 输出当前训练进度和损失。

torch.save(model.state_dict(), "linear_regression_model.pth")  # 保存训练好的模型参数到本地文件。

learned_weight = model.linear.weight.detach().squeeze(0)  # 取出模型学到的两个权重参数。
learned_bias = model.linear.bias.item()  # 取出模型学到的偏置参数。

print("\nModel saved to: linear_regression_model.pth")  # 打印模型保存路径。
print("Learned weight for x1:", round(learned_weight[0].item(), 4))  # 打印模型学到的第一个特征权重。
print("Learned weight for x2:", round(learned_weight[1].item(), 4))  # 打印模型学到的第二个特征权重。
print("Learned bias:", round(learned_bias, 4))  # 打印模型学到的截距。
