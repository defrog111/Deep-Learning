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


class MultiHeadSelfAttention(torch.nn.Module):  # 手写一个最小多头自注意力，方便看清 Q/K/V 和多头拆分。
    def __init__(self, embed_dim: int, num_heads: int) -> None:  # 初始化注意力层；embed_dim 必须能被 num_heads 整除。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.num_heads = num_heads  # 保存注意力头数，后面拆分多头时要用。
        self.head_dim = embed_dim // num_heads  # 计算每个头的维度；这是一个整数。
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)  # 定义 Q 投影层，输入输出 shape 都是 (B, T, D)。
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)  # 定义 K 投影层，输入输出 shape 都是 (B, T, D)。
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)  # 定义 V 投影层，输入输出 shape 都是 (B, T, D)。
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)  # 定义多头拼接后的输出投影层，shape = (B, T, D)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 输入 token 序列 x 的 shape = (B, T, D)。
        batch_size, num_tokens, embed_dim = x.shape  # 读取 batch、大写 token 数和 embedding 维度。
        q = self.q_proj(x)  # 计算 query，shape = (B, T, D)。
        k = self.k_proj(x)  # 计算 key，shape = (B, T, D)。
        v = self.v_proj(x)  # 计算 value，shape = (B, T, D)。
        q = q.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # 拆成多头后 shape = (B, H, T, Dh)。
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # 拆成多头后 shape = (B, H, T, Dh)。
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # 拆成多头后 shape = (B, H, T, Dh)。
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # 计算 scaled dot-product attention 分数，shape = (B, H, T, T)。
        weights = torch.softmax(scores, dim=-1)  # 对最后一维做 softmax，得到注意力权重，shape = (B, H, T, T)。
        context = weights @ v  # 用注意力权重对 value 做加权求和，shape = (B, H, T, Dh)。
        context = context.transpose(1, 2).contiguous().view(batch_size, num_tokens, embed_dim)  # 把多头结果拼回 shape = (B, T, D)。
        out = self.out_proj(context)  # 做输出投影，shape = (B, T, D)。
        return out  # 返回多头自注意力输出。


class TransformerBlock(torch.nn.Module):  # 定义一个最小 Transformer block，包含多头注意力和前馈网络。
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 2) -> None:  # 初始化 block 参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.norm1 = torch.nn.LayerNorm(embed_dim)  # 第一个 LayerNorm，输入输出 shape = (B, T, D)。
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)  # 多头自注意力层，输入输出 shape = (B, T, D)。
        self.norm2 = torch.nn.LayerNorm(embed_dim)  # 第二个 LayerNorm，输入输出 shape = (B, T, D)。
        self.mlp = torch.nn.Sequential(  # 定义最小前馈网络，输入输出 shape = (B, T, D)。
            torch.nn.Linear(embed_dim, embed_dim * mlp_ratio),  # 第一层线性映射，输出 shape = (B, T, 2D)。
            torch.nn.GELU(),  # 非线性激活，shape 不变。
            torch.nn.Linear(embed_dim * mlp_ratio, embed_dim),  # 第二层线性映射，输出 shape = (B, T, D)。
        )  # 结束前馈网络定义。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 输入 token 序列 x 的 shape = (B, T, D)。
        x = x + self.attn(self.norm1(x))  # 先做注意力子层，再做残差连接，shape = (B, T, D)。
        x = x + self.mlp(self.norm2(x))  # 再做前馈子层和残差连接，shape = (B, T, D)。
        return x  # 返回 block 输出。


class TinyViT(torch.nn.Module):  # 定义一个最小 ViT 分类模型。
    def __init__(self, image_size: int = 32, patch_size: int = 8, embed_dim: int = 32, num_classes: int = 10) -> None:  # 初始化 ViT 参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)  # 创建 patch embedding 模块。
        num_patches = (image_size // patch_size) ** 2  # 计算 patch 数量；这是一个整数。
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))  # 定义 class token，shape = (1, 1, embed_dim)。
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # 定义位置编码，shape = (1, num_patches + 1, embed_dim)。
        self.blocks = torch.nn.Sequential(  # 手写堆叠 2 层 Transformer block；输入输出 shape 都是 (B, T, D)。
            TransformerBlock(embed_dim=embed_dim, num_heads=4),
            TransformerBlock(embed_dim=embed_dim, num_heads=4),
        )  # 结束 block 堆叠定义。
        self.norm = torch.nn.LayerNorm(embed_dim)  # 在分类头前再做一次 LayerNorm，shape = (B, T, D)。
        self.head = torch.nn.Linear(embed_dim, num_classes)  # 定义分类头，输入 shape = (B, embed_dim)，输出 shape = (B, num_classes)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 3, 32, 32)。
        x = self.patch_embed(x)  # 得到 patch token 序列，shape = (B, num_patches, embed_dim)。
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # 把 class token 扩展到当前 batch，shape = (B, 1, embed_dim)。
        x = torch.cat([cls_token, x], dim=1)  # 把 class token 拼到 token 序列前面，shape = (B, num_patches + 1, embed_dim)。
        x = x + self.pos_embed  # 加上位置编码，shape 不变。
        x = self.blocks(x)  # 输入手写 Transformer blocks，输出 shape 仍然是 (B, num_patches + 1, embed_dim)。
        x = self.norm(x)  # 做最后一次 LayerNorm，shape 不变。
        cls_feature = x[:, 0]  # 取出 class token 对应的特征向量，shape = (B, embed_dim)。
        logits = self.head(cls_feature)  # 输出分类 logits，shape = (B, num_classes)。
        return logits  # 返回分类结果。


torch.manual_seed(7)  # 固定随机种子，让这个最小训练例子的输出更稳定。
images = torch.randn(4, 3, 32, 32)  # 构造 4 张假图像，shape = (4, 3, 32, 32)。
labels = torch.randint(0, 10, (4,))  # 构造 4 个类别标签，shape = (4,)。
model = TinyViT()  # 创建 ViT 分类模型；模型本身没有 shape。
loss_fn = torch.nn.CrossEntropyLoss()  # 定义分类常用的交叉熵损失。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器，让 ViT 参数根据分类损失更新。

model.train()  # 切到训练模式，后面开始做最小训练循环。
for epoch in range(100):  # 训练 100 轮，让这个小模型先学会当前这 4 张图像的类别标签。
    optimizer.zero_grad()  # 清空上一轮梯度。
    logits = model(images)  # 前向传播得到分类 logits，shape = (4, 10)。
    loss = loss_fn(logits, labels)  # 计算分类损失，输出 shape = ()。
    loss.backward()  # 反向传播，计算各参数梯度。
    optimizer.step()  # 根据梯度更新模型参数。

    if epoch % 25 == 0 or epoch == 99:  # 打印少量轮次，便于观察训练 loss 是否下降。
        print(f"Epoch {epoch:03d} | Train loss: {loss.item():.6f}")

model.eval()  # 切到推理模式，后面的评估和单图 inference 都按 eval 流程来做。
with torch.no_grad():  # 评估和推理阶段不需要梯度。
    logits = model(images)  # 用训练后的模型重新做一次前向传播，shape = (4, 10)。
    loss = loss_fn(logits, labels)  # 计算评估损失，输出 shape = ()。
    pred_labels = torch.argmax(logits, dim=1)  # 把分类 logits 转成预测类别，shape = (4,)。
    accuracy = (pred_labels == labels).float().mean()  # 计算 Accuracy，输出是标量。
    inference_image = images[0].unsqueeze(0)  # 取第 1 张图像做单图 inference，shape = (1, 3, 32, 32)。
    inference_logits = model(inference_image)  # 单张图前向传播，输出 shape = (1, 10)。
    inference_label = torch.argmax(inference_logits, dim=1)  # 单图 inference 得到的预测类别，shape = (1,)。

print("Image shape:", images.shape)  # 打印输入图像 shape。
print("Logits shape:", logits.shape)  # 打印分类输出 shape。
print("Labels shape:", labels.shape)  # 打印标签 shape。
print("Loss shape:", loss.shape)  # 打印损失 shape。
print("Accuracy:", float(accuracy))  # 打印 Accuracy。
print("Inference image shape:", inference_image.shape)  # 打印单张推理图像 shape，说明 inference 也保留 batch 维。
print("Inference logits shape:", inference_logits.shape)  # 打印单张推理 logits 的 shape。
print("Inference label shape:", inference_label.shape)  # 打印单张推理预测类别的 shape。
