import torch  # 导入 PyTorch，用来定义 GRU 模型和张量；这里没有张量 shape。


class GRURegressor(torch.nn.Module):  # 定义一个最小 GRU 回归模型；输入是序列，输出是一个回归值。
    def __init__(self, input_size: int = 1, hidden_size: int = 8) -> None:  # 初始化模型参数；input_size 和 hidden_size 都是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)  # 定义单层 GRU，输入 shape = (B, T, 1)，输出 shape = (B, T, hidden_size)。
        self.head = torch.nn.Linear(hidden_size, 1)  # 定义回归头，输入 shape = (B, hidden_size)，输出 shape = (B, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入序列 x 的 shape = (B, T, 1)。
        out, hidden = self.gru(x)  # 输入 GRU，out 的 shape = (B, T, hidden_size)，hidden 的 shape = (1, B, hidden_size)。
        last_feature = out[:, -1, :]  # 取最后一个时间步特征，shape = (B, hidden_size)。
        pred = self.head(last_feature)  # 输出回归值，shape = (B, 1)。
        return pred  # 返回预测结果。


class ManualGRURegressor(torch.nn.Module):  # 手写一个最小 GRU 回归模型，把 update/reset gate 都显式展开。
    def __init__(self, input_size: int = 1, hidden_size: int = 8) -> None:  # 初始化手写 GRU 所需的各个线性层。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.hidden_size = hidden_size  # 保存隐藏状态维度，后面初始化 h0 时要用到。
        self.z_gate = torch.nn.Linear(input_size + hidden_size, hidden_size)  # update gate 的线性层，输入是 [x_t, h_{t-1}]，shape: (B, 9) -> (B, 8)。
        self.r_gate = torch.nn.Linear(input_size + hidden_size, hidden_size)  # reset gate 的线性层，输入是 [x_t, h_{t-1}]，shape: (B, 9) -> (B, 8)。
        self.n_gate = torch.nn.Linear(input_size + hidden_size, hidden_size)  # 候选隐藏状态的线性层，输入是 [x_t, r_t * h_{t-1}]，shape: (B, 9) -> (B, 8)。
        self.head = torch.nn.Linear(hidden_size, 1)  # 回归头，输入 shape = (B, 8)，输出 shape = (B, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 输入序列 x 的 shape = (B, T, 1)。
        batch_size, seq_len, _ = x.shape  # 读取 batch 大小和时间步长度。
        hidden = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)  # 初始化 h0 为全 0，shape = (B, 8)。

        for time_index in range(seq_len):  # 按时间维一个个取出时间步，这是手写 GRU 最关键的循环。
            current_x = x[:, time_index, :]  # 取第 time_index 个时间步的输入，shape = (B, 1)。
            combined = torch.cat([current_x, hidden], dim=1)  # 把当前输入和上一时刻隐藏状态拼接，shape = (B, 9)。
            z_t = torch.sigmoid(self.z_gate(combined))  # 计算 update gate，shape = (B, 8)。
            r_t = torch.sigmoid(self.r_gate(combined))  # 计算 reset gate，shape = (B, 8)。
            candidate_input = torch.cat([current_x, r_t * hidden], dim=1)  # 把当前输入和重置后的隐藏状态拼接，shape = (B, 9)。
            candidate = torch.tanh(self.n_gate(candidate_input))  # 计算候选隐藏状态，shape = (B, 8)。
            hidden = (1.0 - z_t) * hidden + z_t * candidate  # 按 GRU 公式更新隐藏状态，shape = (B, 8)。

        pred = self.head(hidden)  # 用最后一个时间步的隐藏状态做回归预测，shape = (B, 1)。
        return pred  # 返回预测结果。


class ManualRNNRegressor(torch.nn.Module):  # 手写一个最小 vanilla RNN 回归模型，方便和手写 GRU 对照。
    def __init__(self, input_size: int = 1, hidden_size: int = 8) -> None:  # 初始化手写 RNN 所需的线性层。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.hidden_size = hidden_size  # 保存隐藏状态维度，后面初始化 h0 时要用到。
        self.input_to_hidden = torch.nn.Linear(input_size + hidden_size, hidden_size)  # 把 [x_t, h_{t-1}] 映射到新的隐藏状态，shape: (B, 9) -> (B, 8)。
        self.head = torch.nn.Linear(hidden_size, 1)  # 回归头，输入 shape = (B, 8)，输出 shape = (B, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 输入序列 x 的 shape = (B, T, 1)。
        batch_size, seq_len, _ = x.shape  # 读取 batch 大小和时间步长度。
        hidden = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)  # 初始化 h0 为全 0，shape = (B, 8)。

        for time_index in range(seq_len):  # 按时间维一个个取出时间步，这是手写 RNN 最关键的循环。
            current_x = x[:, time_index, :]  # 取第 time_index 个时间步的输入，shape = (B, 1)。
            combined = torch.cat([current_x, hidden], dim=1)  # 把当前输入和上一时刻隐藏状态拼接，shape = (B, 9)。
            hidden = torch.tanh(self.input_to_hidden(combined))  # 按 vanilla RNN 公式更新隐藏状态，shape = (B, 8)。

        pred = self.head(hidden)  # 用最后一个时间步的隐藏状态做回归预测，shape = (B, 1)。
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
model = GRURegressor(input_size=1, hidden_size=8)  # 创建 GRU 回归模型；模型本身没有 shape。
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

manual_model = ManualGRURegressor(input_size=1, hidden_size=8)  # 创建手写 GRU 模型，和上面高层版保持同样的输入输出规格。
manual_optimizer = torch.optim.Adam(manual_model.parameters(), lr=0.05)  # 为手写 GRU 定义优化器。

for epoch in range(300):  # 同样训练 300 轮，方便和 torch.nn.GRU 版本对照。
    manual_optimizer.zero_grad()  # 清空上一轮梯度。
    manual_predictions = manual_model(sequence_x)  # 手写 GRU 前向传播，shape = (3, 1)。
    manual_loss = loss_fn(manual_predictions, targets)  # 计算当前轮损失，shape = ()。
    manual_loss.backward()  # 反向传播。
    manual_optimizer.step()  # 更新参数。

    if epoch % 100 == 0 or epoch == 299:  # 打印少量轮次，观察损失是否下降。
        print(f"Manual Epoch {epoch:03d} | Loss: {manual_loss.item():.6f}")

with torch.no_grad():  # 推理阶段不需要梯度。
    manual_predictions = manual_model(sequence_x)  # 训练完成后再次预测，shape = (3, 1)。

print("Manual GRU samples and targets:")
for sample, target, pred in zip(sequence_x, targets, manual_predictions):  # 并排打印手写 GRU 的输入、标签和预测值。
    print(sample.squeeze(-1).tolist(), "-> target:", float(target.item()), "| pred:", round(float(pred.item()), 4))

rnn_model = ManualRNNRegressor(input_size=1, hidden_size=8)  # 创建手写 vanilla RNN 模型，和上面 GRU 保持同样的输入输出规格。
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.05)  # 为手写 RNN 定义优化器。

for epoch in range(300):  # 同样训练 300 轮，方便和高层 GRU、手写 GRU 对照。
    rnn_optimizer.zero_grad()  # 清空上一轮梯度。
    rnn_predictions = rnn_model(sequence_x)  # 手写 RNN 前向传播，shape = (3, 1)。
    rnn_loss = loss_fn(rnn_predictions, targets)  # 计算当前轮损失，shape = ()。
    rnn_loss.backward()  # 反向传播。
    rnn_optimizer.step()  # 更新参数。

    if epoch % 100 == 0 or epoch == 299:  # 打印少量轮次，观察损失是否下降。
        print(f"Manual RNN Epoch {epoch:03d} | Loss: {rnn_loss.item():.6f}")

with torch.no_grad():  # 推理阶段不需要梯度。
    rnn_predictions = rnn_model(sequence_x)  # 训练完成后再次预测，shape = (3, 1)。

print("Manual RNN samples and targets:")
for sample, target, pred in zip(sequence_x, targets, rnn_predictions):  # 并排打印手写 RNN 的输入、标签和预测值。
    print(sample.squeeze(-1).tolist(), "-> target:", float(target.item()), "| pred:", round(float(pred.item()), 4))
