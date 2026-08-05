"""
CSV数据处理练习 060：两层GRU时序预测

题目：分别从CSV读取train、val和inference序列，使用两层GRU根据历史窗口预测下一时刻连续值，并在验证集计算MAE与RMSE。

操作过程：
1. 定位时序预测CSV。
2. 读取训练分区。
3. 读取验证分区。
4. 读取推理分区。
5. 定义历史窗口列。
6. 创建训练序列。
7. 创建验证序列。
8. 创建推理序列。
9. 创建训练连续目标。
10. 创建验证连续目标。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
csv_path = Path(__file__).parents[1] / 'data' / 'time_series_sequences.csv'  # 定位时序预测CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 读取训练分区。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 读取验证分区。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 读取推理分区。
steps = [f'step_{index}' for index in range(1, 13)]  # 定义历史窗口列。
train_x = torch.tensor(train_frame[steps].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建训练序列。
val_x = torch.tensor(val_frame[steps].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建验证序列。
inference_x = torch.tensor(inference_frame[steps].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建推理序列。
train_y = torch.tensor(train_frame['next_value'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 创建训练连续目标。
val_y = torch.tensor(val_frame['next_value'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 创建验证连续目标。
x_mean, x_std = train_x.mean(), train_x.std()  # 计算训练输入统计量。
y_mean, y_std = train_y.mean(), train_y.std()  # 计算训练目标统计量。
train_x, val_x, inference_x = (train_x - x_mean) / x_std, (val_x - x_mean) / x_std, (inference_x - x_mean) / x_std  # 转换三个分区。
train_y_scaled = (train_y - y_mean) / y_std  # 标准化训练目标。
class GRUForecaster(nn.Module):  # 定义两层GRU预测器。
    def __init__(self):  # 初始化网络。
        super().__init__()  # 初始化父类。
        self.gru = nn.GRU(1, 16, num_layers=2, batch_first=True)  # 创建两层GRU序列编码器。
        self.output = nn.Linear(16, 1)  # 映射隐藏状态到下一时刻预测。
    def forward(self, values):  # 定义前向传播。
        _, hidden = self.gru(values)  # 获取最终隐藏状态。
        return self.output(hidden[-1])  # 从最后一层状态输出连续值。
torch.manual_seed(42)  # 固定初始化。
model = GRUForecaster()  # 实例化预测器。
criterion = nn.MSELoss()  # 使用MSE训练。
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # 创建优化器。
for epoch in range(400):  # 开始训练阶段。
    model.train()  # 设置训练模式。
    optimizer.zero_grad()  # 清空梯度。
    loss = criterion(model(train_x), train_y_scaled)  # 计算训练损失。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新参数。
model.eval()  # 开始验证阶段。
with torch.no_grad():  # 关闭验证梯度。
    val_predictions = model(val_x) * y_std + y_mean  # 还原验证预测尺度。
    val_mae = (val_predictions - val_y).abs().mean()  # 计算MAE。
    val_rmse = ((val_predictions - val_y) ** 2).mean().sqrt()  # 计算RMSE。
model.eval()  # 开始推理阶段。
with torch.inference_mode():  # 关闭推理梯度。
    inference_predictions = model(inference_x) * y_std + y_mean  # 预测下一时刻值。
assert inference_predictions.shape == (len(inference_frame), 1) and torch.isfinite(val_mae)  # 验证输出和指标。
print('validation_mae:', val_mae.item(), 'validation_rmse:', val_rmse.item(), 'inference_next_value:', inference_predictions.squeeze(1), sep='\n')  # 输出结果。
# 多步预测可使用递归策略反复把预测放回输入，或让模型一次输出整个forecast horizon。
