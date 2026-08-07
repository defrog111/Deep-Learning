"""
CSV数据处理练习 061：Encoder-Decoder Transformer多步预测

题目：分别从CSV读取train、val和inference序列，用前八步作为Encoder输入、后四步作为预测目标，训练两层Encoder和两层Decoder，并在验证与推理阶段自回归生成四个未来值。

操作过程：
1. 定位含十二步序列和固定分区的CSV。
2. 从CSV读取训练序列。
3. 从CSV读取验证序列。
4. 从CSV读取推理序列。
5. 使用前八步作为Encoder历史输入。
6. 使用后四步作为多步预测目标。
7. 记录需要自回归生成的未来步数。
8. 创建shape=(N,8,1)的训练源序列。
9. 创建验证源序列。
10. 创建推理源序列。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取时序CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
csv_path = Path(__file__).parents[1] / 'data' / 'time_series_sequences.csv'  # 定位含十二步序列和固定分区的CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理分区。
source_columns = [f'step_{index}' for index in range(1, 9)]  # 使用前八步作为Encoder历史输入。
target_columns = [f'step_{index}' for index in range(9, 13)]  # 使用后四步作为多步预测目标。
# drop变体：source_frame = train_frame.drop(columns=['sequence_id', *target_columns, 'class_label', 'next_value', 'split'])  # 按列名排除未来目标和非特征字段得到前八步。
# iloc变体：source_frame = train_frame.iloc[:, 1:9]  # 按位置选择step_1到step_8作为Encoder特征。
forecast_horizon = len(target_columns)  # 记录需要自回归生成的未来步数。
train_source = torch.tensor(train_frame[source_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建shape=(N,8,1)的训练源序列。
val_source = torch.tensor(val_frame[source_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建验证源序列。
inference_source = torch.tensor(inference_frame[source_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建推理源序列。
train_target = torch.tensor(train_frame[target_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建shape=(N,4,1)的训练未来目标。
val_target = torch.tensor(val_frame[target_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建只用于验证评分的真实未来序列。
mean, std = train_source.mean(), train_source.std()  # 只用训练历史计算全局标准化参数。
train_source = (train_source - mean) / std  # 标准化训练Encoder输入。
val_source = (val_source - mean) / std  # 使用训练统计量标准化验证Encoder输入。
inference_source = (inference_source - mean) / std  # 使用训练统计量标准化推理Encoder输入。
train_target_scaled = (train_target - mean) / std  # 使用相同尺度转换训练Decoder目标。
class EncoderDecoderForecaster(nn.Module):  # 定义完整Encoder-Decoder Transformer。
    def __init__(self):  # 初始化模型组件。
        super().__init__()  # 初始化nn.Module父类。
        self.source_projection = nn.Linear(1, 16)  # 将Encoder标量输入投影到d_model=16。
        self.target_projection = nn.Linear(1, 16)  # 将Decoder已知值投影到相同特征维度。
        self.source_position = nn.Parameter(torch.zeros(1, len(source_columns), 16))  # 学习八个历史位置的编码。
        self.target_position = nn.Parameter(torch.zeros(1, forecast_horizon, 16))  # 学习四个未来位置的编码。
        self.transformer = nn.Transformer(d_model=16, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=32, dropout=0.0, batch_first=True)  # 创建PyTorch原生两层Encoder和两层Decoder。
        self.output = nn.Linear(16, 1)  # 将Decoder隐藏表示映射为连续预测值。
    def forward(self, source, decoder_input):  # 定义teacher forcing和自回归共用的前向传播。
        source_embedding = self.source_projection(source) + self.source_position[:, :source.size(1)]  # 投影历史值并加入源位置编码。
        target_embedding = self.target_projection(decoder_input) + self.target_position[:, :decoder_input.size(1)]  # 投影Decoder输入并加入目标位置编码。
        target_length = decoder_input.size(1)  # 获取本次Decoder输入长度。
        causal_mask = torch.triu(torch.full((target_length, target_length), float('-inf'), device=decoder_input.device), diagonal=1)  # 创建上三角负无穷因果mask屏蔽未来位置。
        decoded = self.transformer(source_embedding, target_embedding, tgt_mask=causal_mask)  # 先编码历史，再让Decoder进行masked self-attention和cross-attention。
        return self.output(decoded)  # 为每个Decoder位置输出一个连续值。
    def generate(self, source, horizon):  # 定义不读取真实未来值的自回归生成。
        generated = source[:, -1:, :]  # 使用最后一个历史值作为Decoder起始输入。
        for _ in range(horizon):  # 每轮只生成一个新的未来值。
            next_value = self(source, generated)[:, -1:, :]  # 取得当前最后位置预测。
            generated = torch.cat((generated, next_value), dim=1)  # 把预测追加到Decoder输入供下一轮使用。
        return generated[:, 1:, :]  # 去掉起始历史值并返回全部未来预测。
torch.manual_seed(42)  # 固定模型参数初始化。
model = EncoderDecoderForecaster()  # 实例化完整Encoder-Decoder模型。
criterion = nn.MSELoss()  # 使用多步平均均方误差作为训练损失。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 创建Adam优化器。
decoder_start = train_source[:, -1:, :]  # 取每条训练历史的最后值作为Decoder起始token。
teacher_forcing_input = torch.cat((decoder_start, train_target_scaled[:, :-1, :]), dim=1)  # 右移真实目标构造训练Decoder输入。
for epoch in range(500):  # 开始独立训练阶段。
    model.train()  # 切换到训练模式。
    optimizer.zero_grad()  # 清除上一轮梯度。
    train_predictions = model(train_source, teacher_forcing_input)  # 使用teacher forcing并行预测四个未来位置。
    train_loss = criterion(train_predictions, train_target_scaled)  # 对全部未来步计算训练MSE。
    train_loss.backward()  # 反向传播到Encoder、Decoder和投影层。
    optimizer.step()  # 更新全部模型参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证阶段关闭梯度记录。
    val_predictions = model.generate(val_source, forecast_horizon) * std + mean  # 不使用真实未来值自回归生成四步并还原尺度。
    val_mae = (val_predictions - val_target).abs().mean()  # 计算全部样本和未来步的MAE。
    val_rmse = ((val_predictions - val_target) ** 2).mean().sqrt()  # 计算全部未来步RMSE。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 使用推理上下文减少Autograd开销。
    inference_predictions = model.generate(inference_source, forecast_horizon) * std + mean  # 对CSV推理分区自回归预测四个未来值。
inference_result = pd.DataFrame(inference_predictions.squeeze(-1).numpy(), columns=[f'prediction_t+{step}' for step in range(1, forecast_horizon + 1)])  # 把四步Tensor预测转换成可交付表格。
inference_result.insert(0, 'sequence_id', inference_frame['sequence_id'].to_numpy())  # 将预测与原始业务序列ID重新对齐。
assert inference_predictions.shape == (len(inference_frame), forecast_horizon, 1) and inference_result['sequence_id'].is_unique  # 验证多步输出shape和ID唯一性。
print('validation_mae:', val_mae.item(), 'validation_rmse:', val_rmse.item(), 'inference_forecast:', inference_result, sep='\n')  # 输出验证指标和四步推理表。
# 训练时teacher forcing能并行且容易收敛，验证和推理必须自回归，否则使用真实未来值会产生评估泄漏。
# causal mask只屏蔽Decoder未来位置；Decoder仍可通过cross-attention查看全部Encoder历史信息。
# 暴露偏差：训练看到真实前一步而推理看到自身预测，可用scheduled sampling或直接多输出模型缓解。
