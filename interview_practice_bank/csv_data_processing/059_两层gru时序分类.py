"""
CSV数据处理练习 059：两层GRU时序分类

题目：分别从CSV读取train、val和inference序列，使用两层GRU提取顺序信息并根据最后一层隐藏状态完成趋势多分类。

操作过程：
1. 定位时序CSV。
2. 读取训练分区。
3. 读取验证分区。
4. 读取推理分区。
5. 定义有序时间列。
6. 固定类别顺序。
7. 创建标签编码。
8. 创建训练序列。
9. 创建验证序列。
10. 创建推理序列。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
csv_path = Path(__file__).parents[1] / 'data' / 'time_series_sequences.csv'  # 定位时序CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 读取训练分区。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 读取验证分区。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 读取推理分区。
steps = [f'step_{index}' for index in range(1, 13)]  # 定义有序时间列。
classes = ['increasing', 'decreasing', 'seasonal']  # 固定类别顺序。
mapping = {name: index for index, name in enumerate(classes)}  # 创建标签编码。
train_x = torch.tensor(train_frame[steps].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建训练序列。
val_x = torch.tensor(val_frame[steps].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建验证序列。
inference_x = torch.tensor(inference_frame[steps].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 创建推理序列。
train_y = torch.tensor(train_frame['class_label'].map(mapping).to_numpy(), dtype=torch.long)  # 创建训练类别索引。
val_y = torch.tensor(val_frame['class_label'].map(mapping).to_numpy(), dtype=torch.long)  # 创建验证类别索引。
mean, std = train_x.mean(), train_x.std()  # 只从训练序列计算统计量。
train_x, val_x, inference_x = (train_x - mean) / std, (val_x - mean) / std, (inference_x - mean) / std  # 标准化三个分区。
class GRUClassifier(nn.Module):  # 定义两层GRU分类器。
    def __init__(self):  # 初始化网络。
        super().__init__()  # 初始化父类。
        self.gru = nn.GRU(input_size=1, hidden_size=16, num_layers=2, batch_first=True)  # 创建两层GRU。
        self.head = nn.Linear(16, len(classes))  # 映射最终隐藏状态到类别logits。
    def forward(self, values):  # 定义前向传播。
        _, hidden = self.gru(values)  # 获取shape=(层数,N,隐藏维)的最终隐藏状态。
        return self.head(hidden[-1])  # 使用最后一层隐藏状态分类。
torch.manual_seed(42)  # 固定初始化。
model = GRUClassifier()  # 实例化GRU分类器。
criterion = nn.CrossEntropyLoss()  # 使用多分类CE。
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # 创建优化器。
for epoch in range(300):  # 开始训练阶段。
    model.train()  # 切换训练模式。
    optimizer.zero_grad()  # 清空梯度。
    loss = criterion(model(train_x), train_y)  # 计算训练损失。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新参数。
model.eval()  # 开始验证阶段。
with torch.no_grad():  # 关闭验证梯度。
    val_predictions = model(val_x).argmax(1)  # 生成验证类别。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算验证准确率。
model.eval()  # 开始推理阶段。
with torch.inference_mode():  # 关闭推理梯度。
    inference_indices = model(inference_x).argmax(1)  # 预测推理类别索引。
    inference_labels = [classes[index] for index in inference_indices.tolist()]  # 还原字符串类别。
assert len(inference_labels) == len(inference_frame) and val_accuracy.item() >= 0.80  # 验证推理数量和效果。
print('validation_accuracy:', val_accuracy.item(), 'inference_labels:', inference_labels, sep='\n')  # 输出验证与推理结果。
# GRU比LSTM少一个门和一组记忆状态，参数更少、训练更快，但复杂长期依赖上LSTM有时更强。
