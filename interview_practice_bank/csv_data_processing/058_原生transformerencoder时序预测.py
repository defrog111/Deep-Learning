"""
CSV数据处理练习 058：TransformerEncoder递归时序预测

题目：读取时序CSV，按列位置使用TransformerEncoder根据五步历史预测下一连续值，并通过forecast_horizon控制滑动窗口递归预测一步或多步。

操作过程：
1. 读取并清洗时序CSV。
2. 按固定split取得训练、验证和推理分区。
3. 按列位置使用最后五个历史值构造输入窗口。
4. 仅用训练数据计算输入和目标标准化参数。
5. 使用TransformerEncoder训练单步预测模型。
6. 验证时评估真实下一步的MAE和RMSE。
7. 推理时删除窗口最早值并追加本轮预测值。
8. 使用forecast_horizon控制递归预测步数。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
csv_path = Path(__file__).parents[1] / 'data' / 'time_series_sequences.csv'  # 定位时序预测CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建索引。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
window_size = 5  # 指定每次用于预测的历史窗口长度。
history_end_index = 13  # 指定历史数值区域的右边界位置且切片不包含该位置。
history_start_index = history_end_index - window_size  # 根据窗口长度计算最后五个历史值的起始位置。
target_index = 14  # 指定真实下一时刻连续值所在的列位置。
# drop变体：feature_frame = train_frame.drop(columns=['sequence_id', 'class_label', 'next_value', 'split'])  # 也可按列名排除ID、目标和分区列。
# iloc主写法：feature_frame = train_frame.iloc[:, history_start_index:history_end_index]  # 按位置选择长度为五的历史窗口。
forecast_horizon = 4  # 设置为1预测一步，设置为大于1递归预测多步。
train_x = torch.tensor(train_frame.iloc[:, history_start_index:history_end_index].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建shape=(N,5,1)的训练窗口。
val_x = torch.tensor(val_frame.iloc[:, history_start_index:history_end_index].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 按相同位置创建验证窗口。
inference_x = torch.tensor(inference_frame.iloc[:, history_start_index:history_end_index].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 按相同位置创建推理初始窗口。
train_y = torch.tensor(train_frame.iloc[:, target_index].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 按位置创建真实下一值训练目标。
val_y = torch.tensor(val_frame.iloc[:, target_index].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 按位置创建真实下一值验证目标。
x_mean, x_std = train_x.mean(), train_x.std()  # 只从训练历史窗口计算输入统计量。
y_mean, y_std = train_y.mean(), train_y.std()  # 只从训练下一值计算目标统计量。
train_x = (train_x - x_mean) / x_std  # 使用训练统计量标准化训练窗口。
val_x = (val_x - x_mean) / x_std  # 使用相同训练统计量标准化验证窗口。
inference_x = (inference_x - x_mean) / x_std  # 使用相同训练统计量标准化推理窗口。
train_y_scaled = (train_y - y_mean) / y_std  # 标准化训练目标以稳定优化。
class TransformerForecaster(nn.Module):  # 定义单步Transformer预测器。
    def __init__(self):  # 初始化网络组件。
        super().__init__()  # 初始化父类。
        self.input_projection = nn.Linear(1, 16)  # 把每个标量时间点投影到d_model=16。
        self.position = nn.Parameter(torch.zeros(1, window_size, 16))  # 学习五个窗口位置的编码。
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)  # 创建原生Encoder层。
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)  # 堆叠两层TransformerEncoder。
        self.output = nn.Linear(16, 1)  # 从历史表示输出一个标准化下一值。
    def forward(self, values):  # 定义单步前向传播。
        encoded = self.encoder(self.input_projection(values) + self.position)  # 编码当前完整历史窗口。
        return self.output(encoded[:, -1])  # 使用最后位置的上下文预测下一值。
def recursive_forecast(fitted_model, seed_window, horizon):  # 定义滑动窗口递归预测函数。
    current_window = seed_window.clone()  # 复制初始窗口避免原地修改输入。
    generated = []  # 保存每一步原始尺度预测值。
    for _ in range(horizon):  # 根据forecast_horizon逐步生成未来值。
        next_y_scaled = fitted_model(current_window)  # 在目标标准化空间预测下一值。
        next_raw = next_y_scaled * y_std + y_mean  # 把本轮预测还原为原始数值。
        generated.append(next_raw)  # 保存原始尺度预测结果。
        next_x_scaled = (next_raw - x_mean) / x_std  # 把预测值转换到输入窗口使用的标准化尺度。
        current_window = torch.cat((current_window[:, 1:, :], next_x_scaled.unsqueeze(1)), dim=1)  # 删除最早值并把预测值追加到窗口末尾。
    return torch.stack(generated, dim=1)  # 返回shape=(N,horizon,1)的递归预测。
torch.manual_seed(42)  # 固定模型初始化。
model = TransformerForecaster()  # 实例化Transformer预测器。
criterion = nn.MSELoss()  # 使用均方误差训练连续值预测。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 创建Adam优化器。
for epoch in range(400):  # 多轮训练单步预测模型。
    model.train()  # 切换到训练模式。
    optimizer.zero_grad()  # 清除上一轮梯度。
    loss = criterion(model(train_x), train_y_scaled)  # 计算标准化空间的一步预测损失。
    loss.backward()  # 反向传播计算梯度。
    optimizer.step()  # 更新模型参数。
model.eval()  # 切换到验证和推理模式。
with torch.inference_mode():  # 关闭梯度执行验证与递归推理。
    val_predictions = model(val_x) * y_std + y_mean  # 预测验证集真实下一步并还原尺度。
    val_mae = (val_predictions - val_y).abs().mean()  # 计算一步验证MAE。
    val_rmse = ((val_predictions - val_y) ** 2).mean().sqrt()  # 计算一步验证RMSE。
    inference_predictions = recursive_forecast(model, inference_x, forecast_horizon)  # 递归预测指定数量的未来值。
assert inference_predictions.shape == (len(inference_frame), forecast_horizon, 1) and torch.isfinite(val_rmse)  # 验证多步输出shape和指标有效。
print('validation_mae:', val_mae.item(), 'validation_rmse:', val_rmse.item(), 'forecast_horizon:', forecast_horizon, 'future_values:', inference_predictions.squeeze(-1), sep='\n')  # 输出一步指标和多步预测。
# 递归预测会把模型输出继续作为输入，因此远期误差通常会逐步累积。
