import torch  # 导入 PyTorch 主库，用来处理张量计算。
import torch.nn as nn  # 导入神经网络模块，用来定义线性回归模型。


class LinearRegressionModel(nn.Module):  # 定义一个线性回归模型类。
    def __init__(self):  # 初始化模型结构。
        super().__init__()  # 调用父类初始化方法。
        self.linear = nn.Linear(2, 1)  # 定义一个输入维度为 2、输出维度为 1 的线性层。

    def forward(self, x):  # 定义前向传播逻辑。
        return self.linear(x)  # 返回线性层的预测结果。


torch.manual_seed(42)  # 固定随机种子，保证测试时重新生成的数据和训练时一致。

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

# 后 20 个样本作为测试集输入。
X_test = X[train_size:]
# 后 20 个样本作为测试集标签。
y_test = y[train_size:]

print("Test input shape:", X_test.shape)  # 打印测试集输入的形状。
print("Each sample has features:", X.shape[1])  # 打印每个样本包含多少个特征。

model = LinearRegressionModel()  # 创建线性回归模型实例。
model.load_state_dict(torch.load("linear_regression_model.pth"))  # 从本地文件加载训练好的模型参数。
model.eval()  # 将模型切换到评估模式。

criterion = nn.MSELoss()  # 定义均方误差损失函数。

with torch.no_grad():  # 评估时关闭梯度计算，减少开销。
    test_predictions = model(X_test)  # 使用测试集输入得到预测结果。
    test_loss = criterion(test_predictions, y_test)  # 计算测试集上的损失。

print("\nTest Loss:", round(test_loss.item(), 4))  # 打印测试集损失。

sample_input = torch.tensor([[4.0, 1.5]])  # 构造一个包含两个特征的新样本输入用于预测。

with torch.no_grad():  # 关闭梯度计算，进行单样本预测。
    sample_output = model(sample_input)  # 得到输入值为 4.0 时的预测结果。

print("Input:", sample_input.tolist())  # 打印预测时使用的两个输入特征。
print("Predicted output:", round(sample_output.item(), 4))  # 打印模型给出的预测值。
