"""
CSV数据处理练习 049：PyTorch多标签BCE两层FFN

题目：分别从CSV读取train、val和inference数据，用两层FFN同时预测三个可共存标签，为每个标签计算BCE并用独立阈值生成多标签结果。

操作过程：
1. 定位同时含多个0或1标签的CSV。
2. 从CSV读取训练分区。
3. 从CSV读取验证分区。
4. 从CSV读取推理分区。
5. 指定模型输入列。
6. 指定三个彼此独立且可同时为1的标签列。
7. 转换训练特征。
8. 转换验证特征。
9. 转换推理特征。
10. 多标签BCE目标是shape=(N,3)的浮点多热矩阵。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入多标签precision、recall和F1。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位同时含多个0或1标签的CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理分区。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 指定模型输入列。
# drop变体：feature_frame = train_frame.drop(columns=['binary_label', 'class_label', 'label_a', 'label_b', 'label_c', 'split'])  # 按列名排除所有标签和分区列。
# iloc变体：feature_frame = train_frame.iloc[:, :4]  # 按位置选择前四个数值特征。
label_names = ['label_a', 'label_b', 'label_c']  # 指定三个彼此独立且可同时为1的标签列。
train_x = torch.tensor(train_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换训练特征。
val_x = torch.tensor(val_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换验证特征。
inference_x = torch.tensor(inference_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换推理特征。
train_y = torch.tensor(train_frame[label_names].to_numpy(), dtype=torch.float32)  # 多标签BCE目标是shape=(N,3)的浮点多热矩阵。
val_y = torch.tensor(val_frame[label_names].to_numpy(), dtype=torch.float32)  # 准备验证多标签矩阵。
mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True)  # 仅从训练分区学习标准化参数。
train_x, val_x, inference_x = (train_x - mean) / std, (val_x - mean) / std, (inference_x - mean) / std  # 转换三个独立分区。
positive_counts = train_y.sum(dim=0)  # 分别统计每个标签的正样本数。
pos_weight = (len(train_y) - positive_counts) / positive_counts.clamp_min(1)  # 为每个标签计算负样本数与正样本数之比。
torch.manual_seed(42)  # 固定两层FFN初始化。
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(label_names)))  # 创建三个独立logit输出的两层FFN。
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 对三个标签分别计算BCE并补偿正样本比例。
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # 创建Adam优化器。
for epoch in range(500):  # 开始独立训练阶段。
    model.train()  # 切换到训练模式。
    optimizer.zero_grad()  # 清空累积梯度。
    train_logits = model(train_x)  # 输出每个样本的三个独立logit。
    train_loss = criterion(train_logits, train_y)  # 对每个标签计算二元交叉熵后求平均。
    train_loss.backward()  # 反向传播。
    optimizer.step()  # 更新模型参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证时关闭梯度记录。
    val_probabilities = torch.sigmoid(model(val_x))  # 分别得到每个标签的独立概率。
    val_predictions = (val_probabilities >= 0.5).float()  # 每个标签分别应用0.5阈值。
    per_label_accuracy = (val_predictions == val_y).float().mean(dim=0)  # 分别计算三个标签的验证准确率。
val_precision = precision_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各标签precision的宏平均。
val_recall = recall_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各标签recall的宏平均。
val_f1 = f1_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各标签F1的宏平均。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 使用推理上下文关闭梯度。
    inference_probabilities = torch.sigmoid(model(inference_x))  # 生成三个互不排斥的标签概率。
    inference_predictions = (inference_probabilities >= 0.5).to(torch.int64)  # 每行可能得到零个、一个或多个正标签。
assert inference_predictions.shape == (len(inference_frame), len(label_names)) and per_label_accuracy.mean().item() >= 0.75  # 验证多标签输出shape和平均效果。
print('pos_weight:', pos_weight, 'validation_accuracy_per_label:', dict(zip(label_names, per_label_accuracy.tolist())), 'validation_precision_macro:', val_precision, 'validation_recall_macro:', val_recall, 'validation_f1_macro:', val_f1, 'inference_multilabel:', inference_predictions, sep='\n')  # 输出权重、验证分类指标和推理结果。
# 易错点：多标签不是多分类，三个标签可同时为1，因此使用BCE而不是CrossEntropyLoss。
# 阈值变式：可根据每个标签在验证集上的precision与recall分别选择不同阈值，而不是统一使用0.5。
