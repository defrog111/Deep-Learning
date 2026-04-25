import torch  # 导入 PyTorch，用来定义 RNN 模型和张量；这里没有张量 shape。


class VanillaRNNRegressor(torch.nn.Module):  # 定义一个最小 RNN 回归模型；输入是序列，输出是一个回归值。
    def __init__(self, input_size: int = 1, hidden_size: int = 8) -> None:  # 初始化模型参数；input_size 和 hidden_size 都是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)  # 定义单层 RNN，输入 shape = (B, T, 1)，输出 shape = (B, T, hidden_size)。
        self.head = torch.nn.Linear(hidden_size, 1)  # 定义回归头，输入 shape = (B, hidden_size)，输出 shape = (B, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入序列 x 的 shape = (B, T, 1)。
        out, hidden = self.rnn(x)  # 输入 RNN，out 的 shape = (B, T, hidden_size)，hidden 的 shape = (1, B, hidden_size)。
        last_feature = out[:, -1, :]  # 取最后一个时间步特征，shape = (B, hidden_size)。
        pred = self.head(last_feature)  # 输出回归值，shape = (B, 1)。
        return pred  # 返回预测结果。


sequence_x = torch.tensor(  # 构造一个最小时间序列 batch，shape = (3, 4, 1)。
    [
        [[1.0], [2.0], [3.0], [4.0]],
        [[2.0], [3.0], [4.0], [5.0]],
        [[3.0], [4.0], [5.0], [6.0]],
    ],
    dtype=torch.float32,
)  # 结束输入序列定义；整体 shape = (3, 4, 1)。
targets = torch.tensor([[5.0], [6.0], [7.0]], dtype=torch.float32)  # 构造回归目标张量，shape = (3, 1)。
model = VanillaRNNRegressor(input_size=1, hidden_size=8)  # 创建传统 RNN 回归模型；模型本身没有 shape。
predictions = model(sequence_x)  # 前向传播得到预测结果，shape = (3, 1)。
loss_fn = torch.nn.MSELoss()  # 定义均方误差损失函数；loss_fn 本身没有 shape。
loss = loss_fn(predictions, targets)  # 计算 MSE 损失，输出是标量张量，shape = ()。
errors = predictions - targets  # 计算预测误差张量，shape = (3, 1)。
mse = torch.mean(errors ** 2)  # 计算 MSE，输出 shape = ()。
rmse = torch.sqrt(mse)  # 计算 RMSE，输出 shape = ()。
mae = torch.mean(torch.abs(errors))  # 计算 MAE，输出 shape = ()。
ss_res = torch.sum(errors ** 2)  # 计算残差平方和，输出 shape = ()。
ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)  # 计算总平方和，输出 shape = ()。
r2 = 1.0 - ss_res / (ss_tot + 1e-7)  # 计算 R2，输出 shape = ()。

print("Input sequence shape:", sequence_x.shape)  # 打印输入序列 shape。
print("Target shape:", targets.shape)  # 打印目标张量 shape。
print("Prediction shape:", predictions.shape)  # 打印预测张量 shape。
print("Loss shape:", loss.shape)  # 打印损失张量 shape。
print("MSE:", float(mse))  # 打印 MSE。
print("RMSE:", float(rmse))  # 打印 RMSE。
print("MAE:", float(mae))  # 打印 MAE。
print("R2:", float(r2))  # 打印 R2。
