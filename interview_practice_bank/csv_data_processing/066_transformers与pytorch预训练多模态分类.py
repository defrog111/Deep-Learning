"""
CSV数据处理练习 066：Transformers与PyTorch预训练多模态分类

题目：分别从CSV读取train、val和inference商品，使用Hugging Face AutoTokenizer和预训练BERT提取description表示，再用PyTorch融合数值特征完成三分类。

操作过程：
1. 定位同时包含description和数值字段的CSV。
2. 从CSV读取训练文本、数值和标签。
3. 从CSV读取验证文本、数值和标签。
4. 从CSV读取最终推理文本和数值。
5. 选择Google官方轻量预训练BERT以便CPU练习。
6. 使用与bert-tiny共享词表的标准BERT uncased tokenizer。
7. 下载或从缓存加载成熟的BERT WordPiece tokenizer。
8. 下载或从缓存加载预训练Transformer文本编码器。
9. 批量编码训练描述并自动生成attention_mask。
10. 使用同一tokenizer编码验证文本。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取多模态CSV。
import torch  # 导入PyTorch构建数值融合与分类头。
from torch import nn  # 导入神经网络模块。
from transformers import AutoModel, AutoTokenizer  # 导入Hugging Face自动Tokenizer和预训练Transformer。
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入多分类precision、recall和F1。
csv_path = Path(__file__).parents[1] / 'data' / 'multimodal_products.csv'  # 定位同时包含description和数值字段的CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练文本、数值和标签。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证文本、数值和标签。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出最终推理文本和数值。
model_name = 'google/bert_uncased_L-2_H-128_A-2'  # 选择Google官方轻量预训练BERT以便CPU练习。
tokenizer_name = 'bert-base-uncased'  # 使用与bert-tiny共享词表的标准BERT uncased tokenizer。
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)  # 下载或从缓存加载成熟的BERT WordPiece tokenizer。
text_encoder = AutoModel.from_pretrained(model_name)  # 下载或从缓存加载预训练Transformer文本编码器。
train_batch = tokenizer(train_frame['description'].tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')  # 批量编码训练描述并自动生成attention_mask。
val_batch = tokenizer(val_frame['description'].tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')  # 使用同一tokenizer编码验证文本。
inference_batch = tokenizer(inference_frame['description'].tolist(), padding=True, truncation=True, max_length=32, return_tensors='pt')  # 使用同一tokenizer编码推理文本。
for parameter in text_encoder.parameters():  # 遍历预训练文本骨干参数。
    parameter.requires_grad = False  # 冻结BERT并将其作为固定特征提取器。
text_encoder.eval()  # 关闭预训练骨干的dropout以获得稳定文本表示。
def extract_text_features(encoded_batch):  # 定义带attention mask的文本特征提取函数。
    with torch.inference_mode():  # 特征提取不建立计算图以节省内存。
        hidden_states = text_encoder(**encoded_batch).last_hidden_state  # 获得每个有效token的上下文表示。
        valid_mask = encoded_batch['attention_mask'].unsqueeze(-1).to(hidden_states.dtype)  # 将attention mask扩展为池化权重。
        return (hidden_states * valid_mask).sum(1) / valid_mask.sum(1).clamp_min(1)  # 对非padding token执行masked mean pooling。
train_text = extract_text_features(train_batch)  # 一次性提取训练文本的预训练BERT表示。
val_text = extract_text_features(val_batch)  # 提取验证文本表示。
inference_text = extract_text_features(inference_batch)  # 提取推理文本表示。
numeric_columns = ['price', 'rating', 'stock']  # 指定数值模态字段。
# drop变体：numeric_frame = train_frame.drop(columns=['item_id', 'description', 'category', 'split'])  # 按列名排除ID、文本、标签和分区列得到数值特征。
# iloc变体：numeric_frame = train_frame.iloc[:, 1:4]  # 按位置选择price、rating和stock。
train_numeric = torch.tensor(train_frame[numeric_columns].to_numpy(), dtype=torch.float32)  # 创建训练数值Tensor。
val_numeric = torch.tensor(val_frame[numeric_columns].to_numpy(), dtype=torch.float32)  # 创建验证数值Tensor。
inference_numeric = torch.tensor(inference_frame[numeric_columns].to_numpy(), dtype=torch.float32)  # 创建推理数值Tensor。
numeric_mean, numeric_std = train_numeric.mean(0, keepdim=True), train_numeric.std(0, keepdim=True)  # 只用训练分区计算数值标准化参数。
train_numeric = (train_numeric - numeric_mean) / numeric_std  # 标准化训练数值特征。
val_numeric = (val_numeric - numeric_mean) / numeric_std  # 使用训练统计量转换验证数值。
inference_numeric = (inference_numeric - numeric_mean) / numeric_std  # 使用训练统计量转换推理数值。
class_names = ['electronics', 'furniture', 'sports']  # 固定业务类别顺序。
class_to_index = {name: index for index, name in enumerate(class_names)}  # 创建字符串标签到CE类别索引的映射。
train_y = torch.tensor(train_frame['category'].map(class_to_index).to_numpy(), dtype=torch.long)  # 创建训练类别索引。
val_y = torch.tensor(val_frame['category'].map(class_to_index).to_numpy(), dtype=torch.long)  # 创建验证类别索引。
class FusionClassifier(nn.Module):  # 定义预训练文本表示与数值特征的融合分类器。
    def __init__(self):  # 初始化数值分支和融合分类头。
        super().__init__()  # 初始化nn.Module父类。
        self.numeric_encoder = nn.Sequential(nn.Linear(len(numeric_columns), 8), nn.ReLU())  # 将三个数值字段编码成八维表示。
        self.classifier = nn.Sequential(nn.Linear(text_encoder.config.hidden_size + 8, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, len(class_names)))  # 融合BERT表示和数值表示后输出类别logits。
    def forward(self, text_features, numeric_values):  # 定义融合前向传播。
        numeric_features = self.numeric_encoder(numeric_values)  # 编码数值模态。
        fused_features = torch.cat((text_features, numeric_features), dim=1)  # 沿特征维拼接预训练文本和数值表示。
        return self.classifier(fused_features)  # 使用融合特征完成分类。
torch.manual_seed(42)  # 固定融合分类头初始化。
model = FusionClassifier()  # 实例化PyTorch融合模型。
criterion = nn.CrossEntropyLoss()  # 使用互斥三分类交叉熵。
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)  # 使用AdamW训练数值分支和融合头。
for epoch in range(300):  # 开始独立训练阶段。
    model.train()  # 启用融合头训练模式。
    optimizer.zero_grad()  # 清空上一轮梯度。
    train_logits = model(train_text, train_numeric)  # 同时传入BERT文本表示和数值特征。
    train_loss = criterion(train_logits, train_y)  # 计算训练分类损失。
    train_loss.backward()  # 只对数值分支和融合头反向传播。
    optimizer.step()  # 更新可训练参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证阶段关闭梯度。
    val_logits = model(val_text, val_numeric)  # 计算验证类别logits。
    val_predictions = val_logits.argmax(1)  # 选择验证最大logit类别。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算验证准确率。
val_precision = precision_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别precision的宏平均。
val_recall = recall_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别recall的宏平均。
val_f1 = f1_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别F1的宏平均。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 推理阶段关闭Autograd开销。
    inference_probabilities = torch.softmax(model(inference_text, inference_numeric), dim=1)  # 输出推理类别概率。
    inference_indices = inference_probabilities.argmax(1)  # 取得推理类别索引。
    inference_labels = [class_names[index] for index in inference_indices.tolist()]  # 将索引还原为业务类别名称。
inference_result = inference_frame[['item_id', 'description']].assign(predicted_category=inference_labels)  # 将预测结果与CSV商品ID和描述对齐。
assert inference_probabilities.shape == (len(inference_frame), len(class_names)) and val_accuracy.item() >= 0.80  # 验证输出shape和分类效果。
print('backbone:', model_name, 'hidden_size:', text_encoder.config.hidden_size, 'validation_accuracy:', val_accuracy.item(), 'validation_precision_macro:', val_precision, 'validation_recall_macro:', val_recall, 'validation_f1_macro:', val_f1, 'inference:', inference_result, 'probabilities:', inference_probabilities, sep='\n')  # 输出预训练骨干、分类指标和推理结果。
# 冻结骨干适合小数据和CPU快速基线；数据较多时可解冻最后几层并用约1e-5到5e-5的小学习率微调。
# 微调时BERT参数与新分类头通常使用不同学习率，并应通过DataLoader分batch避免一次载入全部文本。
# 首次运行from_pretrained需要网络下载模型；下载完成后Hugging Face会使用本地缓存。
