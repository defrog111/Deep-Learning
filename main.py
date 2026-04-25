import torch  # 导入 PyTorch，用来定义 Transformer 图像分类模型；这里没有张量 shape。


class PatchEmbed(torch.nn.Module):  # 定义 patch embedding，把图像切成 patch 再投影成 token。
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:  # 初始化 patch embedding 模块。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # 用卷积实现 patch 投影，输出 shape = (B, embed_dim, H/patch, W/patch)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, C, H, W)。
        x = self.proj(x)  # 把图像投影成 patch 特征图，shape = (B, embed_dim, Gh, Gw)。
        x = x.flatten(2)  # 把空间维展平，shape = (B, embed_dim, num_patches)。
        x = x.transpose(1, 2)  # 把 token 维放到中间，shape = (B, num_patches, embed_dim)。
        return x  # 返回 patch token 序列。


class TinyViT(torch.nn.Module):  # 定义一个最小 ViT 分类模型。
    def __init__(self, image_size: int = 32, patch_size: int = 8, embed_dim: int = 32, num_classes: int = 10) -> None:  # 初始化 ViT 参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)  # 创建 patch embedding 模块。
        num_patches = (image_size // patch_size) ** 2  # 计算 patch 数量；这是一个整数。
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))  # 定义 class token，shape = (1, 1, embed_dim)。
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # 定义位置编码，shape = (1, num_patches + 1, embed_dim)。
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)  # 定义单层 Transformer encoder；输入输出 shape 都是 (B, T, D)。
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)  # 堆叠 2 层 encoder；输入输出 shape 不变。
        self.head = torch.nn.Linear(embed_dim, num_classes)  # 定义分类头，输入 shape = (B, embed_dim)，输出 shape = (B, num_classes)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 3, 32, 32)。
        x = self.patch_embed(x)  # 得到 patch token 序列，shape = (B, num_patches, embed_dim)。
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # 把 class token 扩展到当前 batch，shape = (B, 1, embed_dim)。
        x = torch.cat([cls_token, x], dim=1)  # 把 class token 拼到 token 序列前面，shape = (B, num_patches + 1, embed_dim)。
        x = x + self.pos_embed  # 加上位置编码，shape 不变。
        x = self.encoder(x)  # 输入 Transformer encoder，输出 shape 仍然是 (B, num_patches + 1, embed_dim)。
        cls_feature = x[:, 0]  # 取出 class token 对应的特征向量，shape = (B, embed_dim)。
        logits = self.head(cls_feature)  # 输出分类 logits，shape = (B, num_classes)。
        return logits  # 返回分类结果。


images = torch.randn(4, 3, 32, 32)  # 构造 4 张假图像，shape = (4, 3, 32, 32)。
labels = torch.randint(0, 10, (4,))  # 构造 4 个类别标签，shape = (4,)。
model = TinyViT()  # 创建 ViT 分类模型；模型本身没有 shape。
logits = model(images)  # 前向传播得到分类 logits，shape = (4, 10)。
loss_fn = torch.nn.CrossEntropyLoss()  # 定义分类常用的交叉熵损失。
loss = loss_fn(logits, labels)  # 计算分类损失，输出 shape = ()。

print(\"Image shape:\", images.shape)  # 打印输入图像 shape。
print(\"Logits shape:\", logits.shape)  # 打印分类输出 shape。
print(\"Labels shape:\", labels.shape)  # 打印标签 shape。
print(\"Loss shape:\", loss.shape)  # 打印损失 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
pred_labels = torch.argmax(logits, dim=1)  # æŠŠåˆ†ç±» logits è½¬æˆé¢„æµ‹ç±»åˆ«ï¼Œshape = (4,)ã€‚
accuracy = (pred_labels == labels).float().mean()  # è®¡ç®— Accuracyï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Accuracy:", float(accuracy))  # æ‰“å° Accuracyã€‚
