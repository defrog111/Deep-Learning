"""
CSV数据处理练习 057：TransformerEncoder递归预测未来类别

题目：读取时序CSV，按列位置把相邻数值转成涨跌类别，使用长度为五的类别窗口预测下一个类别，并通过forecast_horizon控制递归预测一步或多步。

操作过程：
1. 读取并清洗时序CSV。
2. 按固定split取得训练、验证和推理分区。
3. 把相邻时间点转换为decreasing或increasing类别。
4. 使用最近五个方向类别作为输入窗口。
5. 使用下一时刻方向作为监督标签。
6. 使用Embedding和TransformerEncoder完成下一类别预测。
7. 每轮把预测类别追加到窗口并删除最早类别。
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
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入下一方向分类指标。
csv_path = Path(__file__).parents[1] / 'data' / 'time_series_sequences.csv'  # 定位时序CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部序列。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建索引。
train_frame = frame.query("split == 'train'").copy()  # 取得训练分区。
val_frame = frame.query("split == 'val'").copy()  # 取得验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 取得推理分区。
class_names = ['decreasing', 'increasing']  # 定义下一步方向的两个类别。
window_size = 5  # 使用最近五个方向类别预测下一类别。
history_end_index = 13  # 指定历史数值区域的右边界位置且切片不包含该位置。
history_start_index = history_end_index - (window_size + 1)  # 六个连续值产生五个相邻方向类别。
target_index = 14  # 指定真实下一时刻连续值所在的列位置。
# drop变体：feature_frame = train_frame.drop(columns=['sequence_id', 'class_label', 'next_value', 'split'])  # 也可按列名排除ID、目标和分区列。
# iloc主写法：feature_frame = train_frame.iloc[:, history_start_index:history_end_index]  # 按位置选择最后六个历史值。
forecast_horizon = 4  # 设置为1预测一步，设置为大于1递归预测多步。
def make_supervised_data(partition):  # 定义从连续值构造下一方向分类样本的函数。
    values = torch.tensor(partition.iloc[:, history_start_index:history_end_index].to_numpy(), dtype=torch.float32)  # 按位置读取最后六个连续观测值。
    directions = (values[:, 1:] > values[:, :-1]).long()  # 把相邻变化编码为0下降或1上升。
    inputs = directions  # 六个连续值恰好产生长度为五的方向类别窗口。
    next_values = torch.tensor(partition.iloc[:, target_index].to_numpy(), dtype=torch.float32)  # 按位置读取真实下一时刻连续值。
    targets = (next_values > values[:, -1]).long()  # 根据下一值相对最后历史值的变化创建目标类别。
    return inputs, targets  # 返回shape=(N,5)的类别窗口和shape=(N,)的目标。
train_x, train_y = make_supervised_data(train_frame)  # 创建训练分类窗口与下一类别标签。
val_x, val_y = make_supervised_data(val_frame)  # 创建验证分类窗口与下一类别标签。
inference_values = torch.tensor(inference_frame.iloc[:, history_start_index:history_end_index].to_numpy(), dtype=torch.float32)  # 按位置读取推理最后六个历史值。
inference_directions = (inference_values[:, 1:] > inference_values[:, :-1]).long()  # 把推理历史转换为方向类别。
inference_x = inference_directions[:, -window_size:]  # 取得推理所需的最后五个已知类别。
class TransformerNextClass(nn.Module):  # 定义下一类别Transformer模型。
    def __init__(self):  # 初始化模型组件。
        super().__init__()  # 初始化nn.Module父类。
        self.embedding = nn.Embedding(len(class_names), 16)  # 把类别编号转换为可学习向量。
        self.position = nn.Parameter(torch.zeros(1, window_size, 16))  # 为五个窗口位置学习位置编码。
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)  # 创建原生Encoder层。
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)  # 堆叠两层TransformerEncoder。
        self.head = nn.Linear(16, len(class_names))  # 输出下一方向的两个logit。
    def forward(self, class_window):  # 定义单步下一类别预测。
        encoded = self.encoder(self.embedding(class_window) + self.position)  # 编码完整类别窗口及其顺序。
        return self.head(encoded[:, -1])  # 使用最后位置的上下文表示预测下一类别。
    def generate(self, seed_window, horizon):  # 定义可配置步数的递归类别生成。
        current_window = seed_window.clone()  # 复制初始窗口避免修改调用者数据。
        generated = []  # 保存每一步预测的类别编号。
        for _ in range(horizon):  # 按forecast_horizon逐步生成未来类别。
            next_class = self(current_window).argmax(dim=1)  # 预测当前窗口之后的一个类别。
            generated.append(next_class)  # 保存本轮预测。
            current_window = torch.cat((current_window[:, 1:], next_class.unsqueeze(1)), dim=1)  # 删除最早类别并把预测类别追加到末尾。
        return torch.stack(generated, dim=1)  # 返回shape=(N,horizon)的多步类别结果。
torch.manual_seed(42)  # 固定模型初始化以保证结果可复现。
model = TransformerNextClass()  # 实例化下一类别Transformer。
criterion = nn.CrossEntropyLoss()  # 使用互斥分类交叉熵。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 创建Adam优化器。
for epoch in range(300):  # 使用训练窗口多轮优化模型。
    model.train()  # 切换到训练模式。
    optimizer.zero_grad()  # 清除上一轮梯度。
    loss = criterion(model(train_x), train_y)  # 计算一步分类损失。
    loss.backward()  # 通过自动微分计算梯度。
    optimizer.step()  # 更新模型参数。
model.eval()  # 切换到验证和推理模式。
with torch.inference_mode():  # 关闭梯度以执行验证和递归推理。
    val_predictions = model(val_x).argmax(dim=1)  # 预测验证集的下一方向类别。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算一步验证准确率。
    inference_indices = model.generate(inference_x, forecast_horizon)  # 从五类别窗口递归生成未来类别。
val_precision = precision_score(val_y.numpy(), val_predictions.numpy(), zero_division=0)  # 计算上升类precision。
val_recall = recall_score(val_y.numpy(), val_predictions.numpy(), zero_division=0)  # 计算上升类recall。
val_f1 = f1_score(val_y.numpy(), val_predictions.numpy(), zero_division=0)  # 计算上升类F1。
inference_labels = [[class_names[index] for index in row] for row in inference_indices.tolist()]  # 把多步类别编号还原为方向名称。
assert inference_indices.shape == (len(inference_frame), forecast_horizon) and val_accuracy.item() >= 0.75  # 验证多步输出shape和一步分类效果。
print('validation_accuracy:', val_accuracy.item(), 'validation_precision:', val_precision, 'validation_recall:', val_recall, 'validation_f1:', val_f1, 'forecast_horizon:', forecast_horizon, 'future_direction_classes:', inference_labels, sep='\n')  # 输出验证分类指标和递归类别预测。
# 递归多步分类会把自己的预测继续作为输入，因此预测步数越远越容易累积误差。
