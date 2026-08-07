"""
CSV数据处理练习 050：PyTorch加权CE与DataLoader

题目：分别从CSV读取train、val和inference数据，使用TensorDataset与DataLoader进行小批量训练，根据训练类别频数设置加权CE，并在验证集构造混淆矩阵。

操作过程：
1. 定位分类CSV。
2. 从CSV读取训练数据。
3. 从CSV读取验证数据。
4. 从CSV读取推理数据。
5. 指定输入特征列。
6. 固定类别名称和索引顺序。
7. 创建标签编码字典。
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
from torch.utils.data import DataLoader, TensorDataset  # 导入小批量数据封装工具。
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入多分类precision、recall和F1。
csv_path = Path(__file__).parents[1] / 'data' / 'classification_examples.csv'  # 定位分类CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练分区。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证分区。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理分区。
feature_names = ['x1', 'x2', 'x3', 'x4']  # 指定输入特征列。
# drop变体：feature_frame = train_frame.drop(columns=['binary_label', 'class_label', 'label_a', 'label_b', 'label_c', 'split'])  # 按列名排除所有标签和分区列。
# iloc变体：feature_frame = train_frame.iloc[:, :4]  # 按位置选择前四个数值特征。
class_names = ['class_a', 'class_b', 'class_c']  # 固定类别名称和索引顺序。
class_to_index = {name: index for index, name in enumerate(class_names)}  # 创建标签编码字典。
train_x = torch.tensor(train_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换训练特征。
val_x = torch.tensor(val_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换验证特征。
inference_x = torch.tensor(inference_frame[feature_names].to_numpy(), dtype=torch.float32)  # 转换推理特征。
train_y = torch.tensor(train_frame['class_label'].map(class_to_index).to_numpy(), dtype=torch.long)  # 转换训练类别索引。
val_y = torch.tensor(val_frame['class_label'].map(class_to_index).to_numpy(), dtype=torch.long)  # 转换验证类别索引。
mean, std = train_x.mean(0, keepdim=True), train_x.std(0, keepdim=True)  # 仅使用训练特征计算标准化统计量。
train_x, val_x, inference_x = (train_x - mean) / std, (val_x - mean) / std, (inference_x - mean) / std  # 标准化三个分区。
train_dataset = TensorDataset(train_x, train_y)  # 把训练特征和标签封装成可索引数据集。
loader_generator = torch.Generator().manual_seed(42)  # 固定每轮batch打乱顺序。
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, generator=loader_generator)  # 创建小批量训练迭代器。
class_counts = torch.bincount(train_y, minlength=len(class_names)).float()  # 统计训练集中每个类别的样本数。
class_weights = len(train_y) / (len(class_names) * class_counts.clamp_min(1))  # 按频数倒数生成类别权重。
torch.manual_seed(42)  # 固定模型初始化。
model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, len(class_names)))  # 创建两层FFN多分类器。
criterion = nn.CrossEntropyLoss(weight=class_weights)  # 使用加权CE提高少数类别单个样本的损失贡献。
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # 创建Adam优化器。
for epoch in range(250):  # 开始按epoch组织的独立训练阶段。
    model.train()  # 每轮训练前设置训练模式。
    for batch_x, batch_y in train_loader:  # 从DataLoader逐批取得CSV训练样本。
        optimizer.zero_grad()  # 清空上一个batch的梯度。
        batch_loss = criterion(model(batch_x), batch_y)  # 前向传播并计算加权CE。
        batch_loss.backward()  # 反向传播当前batch损失。
        optimizer.step()  # 更新两层FFN参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证阶段不记录梯度。
    val_predictions = model(val_x).argmax(dim=1)  # 获取验证集预测类别。
    confusion_matrix = torch.bincount(val_y * len(class_names) + val_predictions, minlength=len(class_names) ** 2).reshape(len(class_names), len(class_names))  # 不依赖sklearn构造真实类乘预测类的混淆矩阵。
    val_accuracy = confusion_matrix.diag().sum().float() / confusion_matrix.sum()  # 从混淆矩阵对角线计算准确率。
val_precision = precision_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别precision的宏平均。
val_recall = recall_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别recall的宏平均。
val_f1 = f1_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别F1的宏平均。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 关闭推理阶段梯度。
    inference_indices = model(inference_x).argmax(dim=1)  # 对CSV推理分区预测类别索引。
    inference_labels = [class_names[index] for index in inference_indices.tolist()]  # 把索引转换回业务类别名称。
assert confusion_matrix.shape == (3, 3) and len(inference_labels) == len(inference_frame)  # 验证混淆矩阵和推理结果数量。
print('class_counts:', class_counts, 'class_weights:', class_weights, 'validation_accuracy:', val_accuracy.item(), 'validation_precision_macro:', val_precision, 'validation_recall_macro:', val_recall, 'validation_f1_macro:', val_f1, 'confusion_matrix:', confusion_matrix, 'inference_labels:', inference_labels, sep='\n')  # 输出类别权重、验证分类指标和推理类别。
# 常考点：shuffle只用于训练DataLoader，验证和推理通常不打乱以方便结果对应原始行。
# 常考点：严重不平衡任务不能只看accuracy，还应从混淆矩阵计算每类precision、recall和F1。
