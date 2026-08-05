"""
CSV数据处理练习 058：原生TransformerEncoder时序预测

题目：分别从CSV读取train、val和inference序列，使用两层原生TransformerEncoder根据十二个历史值预测下一时刻连续值，并计算MAE与RMSE。

操作过程：
1. 定位时序预测CSV。
2. 从CSV读取训练序列和下一时刻目标。
3. 从CSV读取验证序列和目标。
4. 从CSV读取推理序列。
5. 定义按时间排列的历史窗口列。
6. 创建训练序列Tensor。
7. 创建验证序列Tensor。
8. 创建推理序列Tensor。
9. 创建连续训练目标。
10. 创建连续验证目标。

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
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV读取训练序列和下一时刻目标。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV读取验证序列和目标。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV读取推理序列。
step_columns = [f'step_{index}' for index in range(1, 13)]  # 定义按时间排列的历史窗口列。
train_x = torch.tensor(train_frame[step_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建训练序列Tensor。
val_x = torch.tensor(val_frame[step_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建验证序列Tensor。
inference_x = torch.tensor(inference_frame[step_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建推理序列Tensor。
train_y = torch.tensor(train_frame['next_value'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 创建连续训练目标。
val_y = torch.tensor(val_frame['next_value'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 创建连续验证目标。
x_mean, x_std = train_x.mean(), train_x.std()  # 只从训练历史窗口计算输入统计量。
y_mean, y_std = train_y.mean(), train_y.std()  # 只从训练目标计算输出统计量。
train_x, val_x, inference_x = (train_x - x_mean) / x_std, (val_x - x_mean) / x_std, (inference_x - x_mean) / x_std  # 标准化三个输入分区。
train_y_scaled = (train_y - y_mean) / y_std  # 标准化训练目标以稳定回归优化。
class TransformerForecaster(nn.Module):  # 定义下一时刻预测模型。
    def __init__(self):  # 初始化网络组件。
        super().__init__()  # 初始化父类。
        self.input_projection = nn.Linear(1, 16)  # 投影单变量时间点到d_model。
        self.position = nn.Parameter(torch.zeros(1, len(step_columns), 16))  # 学习每个时间位置的向量。
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)  # 创建原生Encoder层。
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)  # 堆叠两个Encoder层。
        self.output = nn.Linear(16, 1)  # 输出一个标准化的下一时刻预测。
    def forward(self, values):  # 定义前向传播。
        encoded = self.encoder(self.input_projection(values) + self.position)  # 编码完整历史窗口。
        return self.output(encoded[:, -1])  # 使用最后时刻的上下文表示预测下一值。
torch.manual_seed(42)  # 固定模型初始化。
model = TransformerForecaster()  # 实例化两层Transformer预测器。
criterion = nn.MSELoss()  # 使用均方误差训练连续值预测。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 创建优化器。
for epoch in range(400):  # 开始独立训练阶段。
    model.train()  # 设置训练模式。
    optimizer.zero_grad()  # 清空梯度。
    loss = criterion(model(train_x), train_y_scaled)  # 计算标准化空间训练损失。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 关闭验证梯度。
    val_predictions = model(val_x) * y_std + y_mean  # 预测并还原为原始尺度。
    val_mae = (val_predictions - val_y).abs().mean()  # 计算易解释的平均绝对误差。
    val_rmse = ((val_predictions - val_y) ** 2).mean().sqrt()  # 计算对大误差更敏感的RMSE。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 关闭推理梯度。
    inference_predictions = model(inference_x) * y_std + y_mean  # 预测CSV推理序列的下一时刻值。
assert inference_predictions.shape == (len(inference_frame), 1) and torch.isfinite(val_rmse)  # 验证输出shape和指标有效。
print('validation_mae:', val_mae.item(), 'validation_rmse:', val_rmse.item(), 'inference_next_value:', inference_predictions.squeeze(1), sep='\n')  # 输出验证指标和预测结果。
# 面试表达：时间序列必须按时间或实体分区，不能随机把未来记录混入训练造成look-ahead leakage。
# 自回归生成多个未来点时需要因果mask；本例只编码历史窗口并一次预测下一个点，不会看到未来输入。
