"""
CSV数据处理练习 048：PyTorch多分类CE两层FFN

题目：分别从CSV读取train、val和inference数据，把字符串类别编码成整数，用两层FFN和CrossEntropyLoss完成互斥多分类及argmax推理。

操作过程：
1. 定位分类CSV。
2. 从CSV读取训练分区。
3. 从CSV读取验证分区。
4. 从CSV读取推理分区。
5. 指定数值特征列。
6. 固定类别顺序供训练和推理共用。
7. 创建字符串标签到整数的映射。
8. 转换训练特征。
9. 转换验证特征。
10. 转换推理特征。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入多分类precision、recall和F1。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位分类CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理分区。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 指定数值特征列。
# drop变体：feature_frame = train_frame.drop(columns=['binary_label', 'class_label', 'label_a', 'label_b', 'label_c', 'split'])  # 按列名排除所有标签和分区列。
# iloc变体：feature_frame = train_frame.iloc[:, :4]  # 按位置选择前四个数值特征。
class_names = ['class_a', 'class_b', 'class_c']  # 固定类别顺序供训练和推理共用。
class_to_index = {name: index for index, name in enumerate(class_names)}  # 创建字符串标签到整数的映射。
train_x = torch.tensor(train_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换训练特征。
val_x = torch.tensor(val_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换验证特征。
inference_x = torch.tensor(inference_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换推理特征。
train_y = torch.tensor(train_frame['class_label'].map(class_to_index).to_numpy(), dtype=torch.long)  # CE标签必须是shape=(N,)的long类别索引。
val_y = torch.tensor(val_frame['class_label'].map(class_to_index).to_numpy(), dtype=torch.long)  # 准备互斥验证类别索引。
mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True)  # 只用训练数据计算均值和标准差。
train_x, val_x, inference_x = (train_x - mean) / std, (val_x - mean) / std, (inference_x - mean) / std  # 无泄漏地标准化三个分区。
torch.manual_seed(42)  # 固定模型初始化。
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(class_names)))  # 创建两层FFN并为每类输出一个logit。
criterion = nn.CrossEntropyLoss()  # CE内部已经包含LogSoftmax和负对数似然。
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)  # 创建Adam优化器。
for epoch in range(500):  # 开始独立训练阶段。
    model.train()  # 切换为训练模式。
    optimizer.zero_grad()  # 清空梯度。
    train_logits = model(train_x)  # 输出shape=(N,3)的原始logits。
    train_loss = criterion(train_logits, train_y)  # 使用整数类别计算多分类CE损失。
    train_loss.backward()  # 计算梯度。
    optimizer.step()  # 更新模型参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证阶段关闭梯度。
    val_logits = model(val_x)  # 计算验证集三个类别的logits。
    val_predictions = val_logits.argmax(dim=1)  # 选择最大logit对应的唯一类别。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算验证准确率。
val_precision = precision_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别precision的宏平均。
val_recall = recall_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别recall的宏平均。
val_f1 = f1_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别F1的宏平均。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 关闭推理阶段的Autograd开销。
    inference_probabilities = torch.softmax(model(inference_x), dim=1)  # 推理时才把logits转成各类概率。
    inference_indices = inference_probabilities.argmax(dim=1)  # 取得每行概率最大的类别索引。
    inference_labels = [class_names[index] for index in inference_indices.tolist()]  # 把索引还原为CSV中的字符串类别。
assert inference_probabilities.shape == (len(inference_frame), 3) and torch.allclose(inference_probabilities.sum(1), torch.ones(len(inference_frame)))  # 验证概率shape且每行和为1。
print('validation_accuracy:', val_accuracy.item(), 'validation_precision_macro:', val_precision, 'validation_recall_macro:', val_recall, 'validation_f1_macro:', val_f1, 'inference_probability:', inference_probabilities, 'inference_label:', inference_labels, sep='\n')  # 输出验证分类指标和推理结果。
# 易错点：CrossEntropyLoss输入原始logits，目标不是one-hot而是long类型的一维类别索引。
# 易错点：三个类别互斥时用CE；不能为每类独立设阈值，否则可能同时选中多个类别。
