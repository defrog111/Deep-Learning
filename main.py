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
loss_fn = torch.nn.MSELoss()  # 定义均方误差损失函数；loss_fn 本身没有 shape。
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # 定义优化器，让模型参数根据 loss 更新。

for epoch in range(300):  # 训练 300 轮，让模型学习“前 4 个数预测下一个数”。
    optimizer.zero_grad()  # 清空上一轮梯度。
    predictions = model(sequence_x)  # 当前轮前向传播得到预测结果，shape = (3, 1)。
    loss = loss_fn(predictions, targets)  # 计算当前轮 MSE 损失，输出是标量张量，shape = ()。
    loss.backward()  # 反向传播，计算各参数梯度。
    optimizer.step()  # 根据梯度更新参数。

    if epoch % 100 == 0 or epoch == 299:  # 只打印少量轮次，便于观察 loss 下降。
        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}")

with torch.no_grad():  # 推理阶段不需要梯度。
    predictions = model(sequence_x)  # 训练完成后再次预测，shape = (3, 1)。
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
print("Training samples and targets:")
for sample, target, pred in zip(sequence_x, targets, predictions):  # 把每条输入、真实值、预测值并排打印出来。
    print(sample.squeeze(-1).tolist(), "-> target:", float(target.item()), "| pred:", round(float(pred.item()), 4))
print("MSE:", float(mse))  # 打印 MSE。
print("RMSE:", float(rmse))  # 打印 RMSE。
print("MAE:", float(mae))  # 打印 MAE。
print("R2:", float(r2))  # 打印 R2。


class ManualRNNRegressor(torch.nn.Module):  # 手写一个最小 RNN，方便面试时解释每个时间步是怎么循环的。
    def __init__(self, input_size: int = 1, hidden_size: int = 8) -> None:  # 初始化输入到隐藏层、隐藏到隐藏层、隐藏到输出层的映射。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.hidden_size = hidden_size  # 保存隐藏状态维度，后面初始化 h0 时要用到。
        self.x_to_h = torch.nn.Linear(input_size, hidden_size)  # 把当前时间步输入映射到隐藏空间，shape: (B, 1) -> (B, 8)。
        self.h_to_h = torch.nn.Linear(hidden_size, hidden_size)  # 把上一个隐藏状态映射到新的隐藏空间，shape: (B, 8) -> (B, 8)。
        self.head = torch.nn.Linear(hidden_size, 1)  # 用最后一个隐藏状态做回归，shape: (B, 8) -> (B, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 输入序列 x 的 shape = (B, T, 1)。
        batch_size, seq_len, _ = x.shape  # 读取 batch 大小和时间步长度。
        hidden = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)  # 初始化 h0 为全 0，shape = (B, 8)。

        for time_index in range(seq_len):  # 按时间维一个个取出时间步，这是手写 RNN 最关键的循环。
            current_x = x[:, time_index, :]  # 取第 time_index 个时间步的输入，shape = (B, 1)。
            hidden = torch.tanh(self.x_to_h(current_x) + self.h_to_h(hidden))  # 按 h_t = tanh(W_xh x_t + W_hh h_{t-1} + b) 更新隐藏状态。

        pred = self.head(hidden)  # 用最后一个时间步的隐藏状态预测下一个值，shape = (B, 1)。
        return pred  # 返回预测结果。


manual_model = ManualRNNRegressor(input_size=1, hidden_size=8)  # 创建手写 RNN 模型，和上面例子保持同样的输入输出规格。
manual_optimizer = torch.optim.Adam(manual_model.parameters(), lr=0.05)  # 为手写 RNN 定义优化器。

for epoch in range(300):  # 同样训练 300 轮，方便和 torch.nn.RNN 的版本对照。
    manual_optimizer.zero_grad()  # 清空上一轮梯度。
    manual_predictions = manual_model(sequence_x)  # 手写 RNN 前向传播，shape = (3, 1)。
    manual_loss = loss_fn(manual_predictions, targets)  # 计算当前轮损失，shape = ()。
    manual_loss.backward()  # 反向传播。
    manual_optimizer.step()  # 更新参数。

    if epoch % 100 == 0 or epoch == 299:  # 打印少量轮次，观察损失是否下降。
        print(f"Manual Epoch {epoch:03d} | Loss: {manual_loss.item():.6f}")

with torch.no_grad():  # 推理阶段不需要梯度。
    manual_predictions = manual_model(sequence_x)  # 训练完成后再次预测，shape = (3, 1)。

print("Manual RNN samples and targets:")
for sample, target, pred in zip(sequence_x, targets, manual_predictions):  # 并排打印手写 RNN 的输入、标签和预测值。
    print(sample.squeeze(-1).tolist(), "-> target:", float(target.item()), "| pred:", round(float(pred.item()), 4))
