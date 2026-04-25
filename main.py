import torch  # 导入 PyTorch，用来定义多模态融合模型；这里没有张量 shape。


class MultimodalTransformer(torch.nn.Module):  # 定义一个最小多模态 Transformer 分类模型。
    def __init__(self, num_numeric: int = 4, num_categories: int = 20, text_vocab: int = 1000, embed_dim: int = 32, num_classes: int = 3) -> None:  # 初始化模型参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.numeric_proj = torch.nn.Linear(num_numeric, embed_dim)  # 把数值特征投影成 token，输入 shape = (B, 4)，输出 shape = (B, 32)。
        self.category_embed = torch.nn.Embedding(num_categories, embed_dim)  # 把类别特征映射成 embedding，输入 shape = (B,)，输出 shape = (B, 32)。
        self.text_embed = torch.nn.Embedding(text_vocab, embed_dim)  # 把文本 token id 映射成 embedding，输入 shape = (B, T_text)，输出 shape = (B, T_text, 32)。
        self.image_patch = torch.nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)  # 把图像切成 patch token，输出 shape = (B, 32, Gh, Gw)。
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))  # 定义分类 token，shape = (1, 1, 32)。
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)  # 定义 Transformer encoder layer。
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)  # 堆叠两层 encoder，输入输出 shape 都是 (B, T, 32)。
        self.head = torch.nn.Linear(embed_dim, num_classes)  # 定义分类头，输入 shape = (B, 32)，输出 shape = (B, 3)。

    def forward(self, numeric_x: torch.Tensor, category_x: torch.Tensor, image_x: torch.Tensor, text_x: torch.Tensor) -> torch.Tensor:  # 定义前向传播。
        numeric_token = self.numeric_proj(numeric_x).unsqueeze(1)  # 把数值特征投影成一个 token，shape = (B, 1, 32)。
        category_token = self.category_embed(category_x).unsqueeze(1)  # 把类别特征映射成一个 token，shape = (B, 1, 32)。
        text_tokens = self.text_embed(text_x)  # 把文本 token id 变成文本 token 序列，shape = (B, T_text, 32)。
        image_tokens = self.image_patch(image_x)  # 把图像投影成 patch 特征图，shape = (B, 32, Gh, Gw)。
        image_tokens = image_tokens.flatten(2).transpose(1, 2)  # 展平成图像 token 序列，shape = (B, T_image, 32)。
        cls_token = self.cls_token.expand(numeric_x.size(0), -1, -1)  # 把分类 token 扩展到当前 batch，shape = (B, 1, 32)。
        tokens = torch.cat([cls_token, numeric_token, category_token, image_tokens, text_tokens], dim=1)  # 把多种模态 token 拼成一个总序列，shape = (B, T_total, 32)。
        encoded = self.encoder(tokens)  # 输入 Transformer encoder，输出 shape = (B, T_total, 32)。
        cls_feature = encoded[:, 0]  # 取分类 token 对应的融合特征，shape = (B, 32)。
        logits = self.head(cls_feature)  # 输出分类 logits，shape = (B, 3)。
        return logits  # 返回多模态分类结果。


numeric_x = torch.randn(2, 4)  # 构造数值特征矩阵，shape = (2, 4)。
category_x = torch.tensor([3, 7], dtype=torch.long)  # 构造类别特征向量，shape = (2,)。
image_x = torch.randn(2, 3, 32, 32)  # 构造图像张量，shape = (2, 3, 32, 32)。
text_x = torch.randint(0, 1000, (2, 5))  # 构造文本 token id 矩阵，shape = (2, 5)。
labels = torch.randint(0, 3, (2,))  # 构造分类标签向量，shape = (2,)。
model = MultimodalTransformer()  # 创建多模态 Transformer 模型；模型本身没有 shape。
logits = model(numeric_x, category_x, image_x, text_x)  # 前向传播得到分类 logits，shape = (2, 3)。
loss_fn = torch.nn.CrossEntropyLoss()  # 定义多分类交叉熵损失。
loss = loss_fn(logits, labels)  # 计算分类损失，输出 shape = ()。

print(\"Numeric shape:\", numeric_x.shape)  # 打印数值特征 shape。
print(\"Category shape:\", category_x.shape)  # 打印类别特征 shape。
print(\"Image shape:\", image_x.shape)  # 打印图像张量 shape。
print(\"Text shape:\", text_x.shape)  # 打印文本 token 矩阵 shape。
print(\"Logits shape:\", logits.shape)  # 打印分类输出 shape。
print(\"Loss shape:\", loss.shape)  # 打印损失 shape。
