"""
CSV数据处理练习 047：PyTorch二分类BCE两层FFN

题目：分别从CSV读取train、val和inference数据，用两层FFN、BCEWithLogitsLoss训练二分类器，在验证集选择阈值并对推理集输出概率与类别。

操作过程：
1. 定位带固定数据分区的分类CSV。
2. 从CSV取出训练数据。
3. 从CSV独立取出验证数据。
4. 从CSV独立取出最终推理数据。
5. 定义四个数值输入特征。
6. 转换训练特征为二维Tensor。
7. 转换验证特征为二维Tensor。
8. 转换推理特征为二维Tensor。
9. BCE目标必须为浮点且与输出同为(N,1)。
10. 准备验证标签但不参与参数更新。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位带固定数据分区的分类CSV。
train_frame = pd.read_csv(csv_path).query("split == 'train'").copy()  # 从CSV取出训练数据。
val_frame = pd.read_csv(csv_path).query("split == 'val'").copy()  # 从CSV独立取出验证数据。
inference_frame = pd.read_csv(csv_path).query("split == 'inference'").copy()  # 从CSV独立取出最终推理数据。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 定义四个数值输入特征。
train_x = torch.tensor(train_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换训练特征为二维Tensor。
val_x = torch.tensor(val_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换验证特征为二维Tensor。
inference_x = torch.tensor(inference_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换推理特征为二维Tensor。
train_y = torch.tensor(train_frame['binary_label'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # BCE目标必须为浮点且与输出同为(N,1)。
val_y = torch.tensor(val_frame['binary_label'].to_numpy(), dtype=torch.float32).unsqueeze(1)  # 准备验证标签但不参与参数更新。
mean, std = train_x.mean(dim=0, keepdim=True), train_x.std(dim=0, keepdim=True)  # 只用训练集计算标准化统计量。
train_x = (train_x - mean) / std  # 标准化训练输入。
val_x = (val_x - mean) / std  # 使用训练统计量标准化验证输入。
inference_x = (inference_x - mean) / std  # 使用训练统计量标准化推理输入。
torch.manual_seed(42)  # 固定两层FFN的参数初始化。
model = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 1))  # 创建输入到隐藏层再到单logit输出的两层FFN。
criterion = nn.BCEWithLogitsLoss()  # 把稳定的Sigmoid计算和二元交叉熵合在一起。
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # 创建Adam优化器。
for epoch in range(400):  # 开始独立训练阶段。
    model.train()  # 启用训练模式。
    optimizer.zero_grad()  # 清空上一轮梯度。
    train_logits = model(train_x)  # 前向传播输出未经Sigmoid的logits。
    train_loss = criterion(train_logits, train_y)  # 用训练标签计算BCE损失。
    train_loss.backward()  # 反向传播计算梯度。
    optimizer.step()  # 更新两层FFN参数。
model.eval()  # 开始独立验证阶段并切换为评估模式。
with torch.no_grad():  # 验证时关闭梯度记录。
    val_probabilities = torch.sigmoid(model(val_x))  # 把验证logits转换成正类概率。
    val_predictions = (val_probabilities >= 0.5).float()  # 使用0.5阈值得到二分类结果。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算验证准确率。
model.eval()  # 开始独立推理阶段并保持评估模式。
with torch.inference_mode():  # 使用专门的推理上下文进一步减少开销。
    inference_probabilities = torch.sigmoid(model(inference_x))  # 输出推理数据属于正类的概率。
    inference_predictions = (inference_probabilities >= 0.5).to(torch.int64)  # 根据阈值生成最终0或1类别。
assert inference_predictions.shape == (len(inference_frame), 1) and val_accuracy.item() >= 0.70  # 验证输出shape及验证效果。
print('validation_accuracy:', val_accuracy.item(), 'inference_probability:', inference_probabilities.squeeze(1), 'inference_class:', inference_predictions.squeeze(1), sep='\n')  # 输出验证指标和推理结果。
# 易错点：BCEWithLogitsLoss直接接logits，训练前不要再手动调用sigmoid。
# 另一种写法：二分类也能输出两个logits并使用CrossEntropyLoss，但单logit加BCE更直接。
