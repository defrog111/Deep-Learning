"""
CSV数据处理练习 062：Transformer文本与数值多模态分类

题目：分别从CSV读取train、val和inference商品，使用原生TransformerEncoder编码description文本、MLP编码价格评分库存，再融合两种模态完成三分类。

操作过程：
1. 定位多模态商品CSV。
2. 从CSV读取训练文本、数值和标签。
3. 从CSV读取验证文本、数值和标签。
4. 从CSV读取推理文本和数值。
5. 定义数值模态字段。
6. 固定三个互斥商品类别。
7. 创建类别字符串到整数索引的映射。
8. 只统计训练文本词频以避免验证和推理词汇泄漏。
9. 预留padding和未知词索引。
10. 以稳定顺序加入训练集词语。

完成标准：
- 必须使用pd.read_csv从磁盘载入CSV。
- 脚本能够独立运行且assert全部通过。
- 先自己实现，再对照下面逐行中文注释的参考代码。
"""

from collections import Counter  # 导入词频统计工具以建立训练词表。
from pathlib import Path  # 导入跨平台路径工具。
import pandas as pd  # 导入Pandas读取包含文本和数值的CSV。
import torch  # 导入PyTorch。
from torch import nn  # 导入神经网络模块。
from sklearn.metrics import f1_score, precision_score, recall_score  # 导入多分类precision、recall和F1。
csv_path = Path(__file__).parents[1] / 'data' / 'multimodal_products.csv'  # 定位多模态商品CSV。
frame = pd.read_csv(csv_path)  # 从CSV读取全部数据。
frame = frame.dropna(how='all').reset_index(drop=True)  # 删除整行全为空的无效记录并重建连续索引。
train_frame = frame.query("split == 'train'").copy()  # 从清洗后的数据取出训练文本、数值和标签。
val_frame = frame.query("split == 'val'").copy()  # 从清洗后的数据取出验证文本、数值和标签。
inference_frame = frame.query("split == 'inference'").copy()  # 从清洗后的数据取出推理文本和数值。
numeric_columns = ['price', 'rating', 'stock']  # 定义数值模态字段。
# drop变体：numeric_frame = train_frame.drop(columns=['item_id', 'description', 'category', 'split'])  # 按列名排除ID、文本、标签和分区列得到数值特征。
# iloc变体：numeric_frame = train_frame.iloc[:, 1:4]  # 按位置选择price、rating和stock。
class_names = ['electronics', 'furniture', 'sports']  # 固定三个互斥商品类别。
class_to_index = {name: index for index, name in enumerate(class_names)}  # 创建类别字符串到整数索引的映射。
token_counts = Counter(token for text in train_frame['description'] for token in text.lower().split())  # 只统计训练文本词频以避免验证和推理词汇泄漏。
vocabulary = {'<pad>': 0, '<unk>': 1}  # 预留padding和未知词索引。
vocabulary.update({token: index for index, token in enumerate(sorted(token_counts), start=2)})  # 以稳定顺序加入训练集词语。
maximum_length = 8  # 固定文本序列长度供批量Transformer计算。
def encode_texts(texts):  # 定义文本分词、截断和补齐函数。
    rows = []  # 准备保存每条描述的token索引。
    for text in texts:  # 逐条处理CSV描述文本。
        token_ids = [vocabulary.get(token, vocabulary['<unk>']) for token in text.lower().split()[:maximum_length]]  # 将训练已知词映射为索引并用unk处理新词。
        padded_ids = token_ids + [vocabulary['<pad>']] * (maximum_length - len(token_ids))  # 在句尾补pad到固定长度。
        rows.append(padded_ids)  # 保存编码后的单条描述。
    return torch.tensor(rows, dtype=torch.long)  # 返回shape=(N,文本长度)的整数Tensor。
train_tokens = encode_texts(train_frame['description'])  # 编码训练文本模态。
val_tokens = encode_texts(val_frame['description'])  # 用训练词表编码验证文本。
inference_tokens = encode_texts(inference_frame['description'])  # 用训练词表编码推理文本并安全处理未知词。
train_numeric = torch.tensor(train_frame[numeric_columns].to_numpy(), dtype=torch.float32)  # 创建训练数值模态Tensor。
val_numeric = torch.tensor(val_frame[numeric_columns].to_numpy(), dtype=torch.float32)  # 创建验证数值模态Tensor。
inference_numeric = torch.tensor(inference_frame[numeric_columns].to_numpy(), dtype=torch.float32)  # 创建推理数值模态Tensor。
numeric_mean, numeric_std = train_numeric.mean(0, keepdim=True), train_numeric.std(0, keepdim=True)  # 只用训练数值计算标准化参数。
train_numeric = (train_numeric - numeric_mean) / numeric_std  # 标准化训练数值模态。
val_numeric = (val_numeric - numeric_mean) / numeric_std  # 使用训练统计量转换验证数值。
inference_numeric = (inference_numeric - numeric_mean) / numeric_std  # 使用训练统计量转换推理数值。
train_y = torch.tensor(train_frame['category'].map(class_to_index).to_numpy(), dtype=torch.long)  # 创建训练类别索引。
val_y = torch.tensor(val_frame['category'].map(class_to_index).to_numpy(), dtype=torch.long)  # 创建验证类别索引。
class MultimodalTransformer(nn.Module):  # 定义文本与数值融合分类器。
    def __init__(self):  # 初始化各模态分支和融合层。
        super().__init__()  # 初始化nn.Module父类。
        self.embedding = nn.Embedding(len(vocabulary), 16, padding_idx=vocabulary['<pad>'])  # 将文本token映射为16维向量并固定pad嵌入。
        self.position = nn.Parameter(torch.zeros(1, maximum_length, 16))  # 学习文本token位置编码。
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0, batch_first=True)  # 创建PyTorch原生文本Encoder层。
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 堆叠两层TransformerEncoder处理描述。
        self.numeric_encoder = nn.Sequential(nn.Linear(len(numeric_columns), 8), nn.ReLU())  # 用小型数值分支提取价格评分库存特征。
        self.fusion_head = nn.Sequential(nn.Linear(16 + 8, 16), nn.ReLU(), nn.Linear(16, len(class_names)))  # 拼接两个模态后输出三个类别logit。
    def forward(self, token_ids, numeric_values):  # 定义多模态前向传播。
        padding_mask = token_ids.eq(vocabulary['<pad>'])  # 标记文本中的pad位置供Attention忽略。
        text_encoded = self.text_encoder(self.embedding(token_ids) + self.position, src_key_padding_mask=padding_mask)  # 用attention mask编码有效文本token。
        valid_tokens = (~padding_mask).unsqueeze(-1)  # 构造有效token池化权重。
        text_features = (text_encoded * valid_tokens).sum(1) / valid_tokens.sum(1).clamp_min(1)  # 对非pad位置进行masked mean pooling。
        numeric_features = self.numeric_encoder(numeric_values)  # 编码标准化数值模态。
        fused_features = torch.cat((text_features, numeric_features), dim=1)  # 在特征维拼接文本和数值表示。
        return self.fusion_head(fused_features)  # 使用融合表示输出分类logits。
torch.manual_seed(42)  # 固定模型初始化。
model = MultimodalTransformer()  # 实例化两层文本Transformer和数值融合模型。
model.text_encoder.enable_nested_tensor = False  # 关闭实验性nested tensor优化并保留标准padding mask计算。
model.text_encoder.use_nested_tensor = False  # 明确让当前PyTorch版本的Encoder走稳定普通Tensor路径。
criterion = nn.CrossEntropyLoss()  # 使用互斥三分类交叉熵。
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  # 创建Adam优化器。
for epoch in range(350):  # 开始独立训练阶段。
    model.train()  # 切换为训练模式。
    optimizer.zero_grad()  # 清空上一轮梯度。
    train_logits = model(train_tokens, train_numeric)  # 同时向模型传入训练文本和数值。
    train_loss = criterion(train_logits, train_y)  # 计算融合分类损失。
    train_loss.backward()  # 反向传播到两个模态分支。
    optimizer.step()  # 更新Transformer、数值编码器和融合层参数。
model.eval()  # 开始独立验证阶段。
with torch.no_grad():  # 验证时关闭梯度。
    val_predictions = model(val_tokens, val_numeric).argmax(1)  # 生成验证类别索引。
    val_accuracy = (val_predictions == val_y).float().mean()  # 计算验证准确率。
val_precision = precision_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别precision的宏平均。
val_recall = recall_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别recall的宏平均。
val_f1 = f1_score(val_y.numpy(), val_predictions.numpy(), average='macro', zero_division=0)  # 计算各类别F1的宏平均。
model.eval()  # 开始独立推理阶段。
with torch.inference_mode():  # 使用推理上下文减少开销。
    inference_probabilities = torch.softmax(model(inference_tokens, inference_numeric), dim=1)  # 输出推理商品的三类概率。
    inference_indices = inference_probabilities.argmax(1)  # 选择概率最大的类别。
    inference_labels = [class_names[index] for index in inference_indices.tolist()]  # 将类别索引还原为名称。
inference_result = inference_frame[['item_id', 'description']].assign(predicted_category=inference_labels)  # 将多模态预测与商品ID和描述对齐。
assert inference_probabilities.shape == (len(inference_frame), len(class_names)) and val_accuracy.item() >= 0.80  # 验证概率shape和分类效果。
print('vocabulary_size:', len(vocabulary), 'validation_accuracy:', val_accuracy.item(), 'validation_precision_macro:', val_precision, 'validation_recall_macro:', val_recall, 'validation_f1_macro:', val_f1, 'inference_result:', inference_result, 'probabilities:', inference_probabilities, sep='\n')  # 输出词表、验证分类指标和推理结果。
# 这是late/intermediate fusion：先分别编码文本和数值，再拼接表示；early fusion则更早把模态转成统一token。
# 易错点：词表、数值均值和标准差都只能从train建立，validation与inference的新词必须映射为unk。
# 实际项目可把手写词表替换为预训练BERT tokenizer和encoder，但CSV读取、数值分支与融合流程保持一致。
