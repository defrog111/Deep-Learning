"""
CSV数据处理练习 057：原生TransformerEncoder时序分类

题目：分别从CSV读取train、val和inference序列，使用输入投影、可学习位置编码和两层原生TransformerEncoder完成趋势多分类。

操作过程：
1. 定位宽表形式的时序CSV。
2. 从CSV读取训练序列。
3. 从CSV读取验证序列。
4. 从CSV读取推理序列。
5. 按时间顺序定义十二个观测列。
6. 固定三种互斥趋势类别。
7. 创建字符串标签编码。
8. 转换为(N,时间,特征)序列Tensor。
9. 转换验证序列。
10. 转换推理序列。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取时序CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
csv_path = Path(__file__).parents[1] / 'data' / 'time_series_sequences.csv'  # 定位宽表形式的时序CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV读取训练序列。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV读取验证序列。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV读取推理序列。
step_columns = [f'step_{index}' for index in range(1, 13)]  # 按时间顺序定义十二个观测列。
class_names = ['increasing', 'decreasing', 'seasonal']  # 固定三种互斥趋势类别。
class_to_index = {name: index for index, name in enumerate(class_names)}  # 创建字符串标签编码。
train_x = torch.tensor(train_frame[step_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 转换为(N,时间,特征)序列Tensor。
val_x = torch.tensor(val_frame[step_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 转换验证序列。
inference_x = torch.tensor(inference_frame[step_columns].to_numpy(), dtype=torch.float32).unsqueeze(-1)  # 转换推理序列。
train_y = torch.tensor(train_frame['class_label'].map(class_to_index).to_numpy(), dtype=torch.long)  # 创建CE需要的一维long标签。
val_y = torch.tensor(val_frame['class_label'].map(class_to_index).to_numpy(), dtype=torch.long)  # 创建验证类别索引。
mean, std = train_x.mean(), train_x.std()  # 只用训练序列计算全局标准化参数。
train_x, val_x, inference_x = (train_x - mean) / std, (val_x - mean) / std, (inference_x - mean) / std  # 无泄漏地转换三个分区。
class TransformerClassifier(nn.Module):  # 定义原生Transformer时序分类器。
    def __init__(self):  # 初始化模型组件。
        super().__init__()  # 初始化nn.Module父类。
        self.input_projection = nn.Linear(1, 16)  # 把每个标量观测映射到d_model=16。
        self.position = nn.Parameter(torch.zeros(1, len(step_columns), 16))  # 为十二个时刻学习不同位置向量。
        layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)  # 创建PyTorch原生Encoder层。
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)  # 堆叠两层TransformerEncoder。
        self.head = nn.Linear(16, len(class_names))  # 把池化表示映射为三个类别logit。
    def forward(self, values):  # 定义前向传播。
        encoded = self.encoder(self.input_projection(values) + self.position)  # 加入位置信息后执行自注意力编码。
        return self.head(encoded.mean(dim=1))  # 平均池化全部时刻并输出分类logits。
torch.manual_seed(42)  # 固定模型初始化。
model = TransformerClassifier()  # 实例化两层原生Transformer分类器。
criterion = nn.CrossEntropyLoss()  # 使用互斥多分类交叉熵。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 创建Adam优化器。
for epoch in range(300):  # 开始独立训练阶段。
    model.train()  # 切换到训练模式。
    optimizer.zero_grad()  # 清空梯度。
    loss = criterion(model(train_x), train_y)  # 前向传播并计算训练CE。
    loss.backward()  # 反向传播。
    optimizer.step()  # 更新Transformer参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证时关闭梯度。
    val_predictions = model(val_x).argmax(dim=1)  # 获取验证趋势类别。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算验证准确率。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 使用推理上下文。
    inference_probabilities = torch.softmax(model(inference_x), dim=1)  # 把推理logits转换为类别概率。
    inference_labels = [class_names[index] for index in inference_probabilities.argmax(1).tolist()]  # 将最大概率索引还原为趋势名称。
assert inference_probabilities.shape == (len(inference_frame), 3) and val_accuracy.item() >= 0.80  # 验证输出shape和分类效果。
print('validation_accuracy:', val_accuracy.item(), 'inference_labels:', inference_labels, 'probabilities:', inference_probabilities, sep='\n')  # 输出验证和推理结果。
# 本例序列等长所以不需要padding mask；变长序列必须向encoder传src_key_padding_mask屏蔽pad位置。
# 易错点：Transformer本身不知道时间顺序，必须加入位置编码或其他时间位置特征。
