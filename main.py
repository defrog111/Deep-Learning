import torch  # 导入 PyTorch，用来定义 Transformer 图像去噪模型；这里没有张量 shape。


class PatchEmbed(torch.nn.Module):  # 定义 patch embedding 模块，把图像变成 patch token。
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:  # 初始化 patch embedding 参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)  # 用卷积实现 patch 投影，输出 shape = (B, embed_dim, Gh, Gw)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, 1, H, W)。
        x = self.proj(x)  # 投影成 patch 特征图，shape = (B, embed_dim, Gh, Gw)。
        x = x.flatten(2).transpose(1, 2)  # 展平并转置成 token 序列，shape = (B, num_patches, embed_dim)。
        return x  # 返回 patch token 序列。


class PatchUnembed(torch.nn.Module):  # 定义 patch unembedding 模块，把 token 序列还原成图像。
    def __init__(self, embed_dim: int, out_channels: int, patch_grid: int, patch_size: int) -> None:  # 初始化反投影参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.patch_grid = patch_grid  # 保存 patch 网格尺寸，比如 4x4。
        self.patch_size = patch_size  # 保存 patch 大小，比如 8。
        self.out_channels = out_channels  # 保存输出通道数，比如 1。
        self.linear = torch.nn.Linear(embed_dim, out_channels * patch_size * patch_size)  # 把每个 token 线性映射回一个 patch 像素块。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, num_patches, embed_dim)。
        x = self.linear(x)  # 把每个 token 映射成 patch 像素块，shape = (B, num_patches, patch_area)。
        b = x.size(0)  # 取出 batch size，是整数。
        x = x.view(b, self.patch_grid, self.patch_grid, self.out_channels, self.patch_size, self.patch_size)  # 重排成 6 维 patch 网格，shape = (B, Gh, Gw, C, Ph, Pw)。
        x = x.permute(0, 3, 1, 4, 2, 5)  # 调整维度顺序，shape = (B, C, Gh, Ph, Gw, Pw)。
        x = x.reshape(b, self.out_channels, self.patch_grid * self.patch_size, self.patch_grid * self.patch_size)  # 拼回整张图像，shape = (B, C, H, W)。
        return x  # 返回重建后的图像。


class TransformerDenoiser(torch.nn.Module):  # 定义一个最小 Transformer 图像去噪模型。
    def __init__(self, image_size: int = 32, patch_size: int = 8, embed_dim: int = 32) -> None:  # 初始化模型参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.patch_embed = PatchEmbed(1, embed_dim, patch_size)  # 定义 patch embedding，输入是单通道灰度图。
        num_patches = (image_size // patch_size) ** 2  # 计算 patch 个数；这是整数。
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # 定义位置编码，shape = (1, num_patches, embed_dim)。
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)  # 定义 encoder layer。
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)  # 定义两层 Transformer encoder。
        self.patch_unembed = PatchUnembed(embed_dim, 1, image_size // patch_size, patch_size)  # 定义 patch 反投影模块。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入带噪图像 shape = (B, 1, 32, 32)。
        x = self.patch_embed(x)  # 转成 token 序列，shape = (B, num_patches, embed_dim)。
        x = x + self.pos_embed  # 加位置编码，shape 不变。
        x = self.encoder(x)  # 输入 Transformer encoder，输出 shape 不变。
        x = self.patch_unembed(x)  # 还原成图像，输出 shape = (B, 1, 32, 32)。
        return x  # 返回去噪后的图像。


noisy_images = torch.randn(2, 1, 32, 32)  # 构造 2 张带噪灰度图，shape = (2, 1, 32, 32)。
clean_images = torch.randn(2, 1, 32, 32)  # 构造 2 张干净目标图，shape = (2, 1, 32, 32)。
model = TransformerDenoiser()  # 创建 Transformer 去噪模型；模型本身没有 shape。
denoised_images = model(noisy_images)  # 前向传播得到去噪结果，shape = (2, 1, 32, 32)。
loss_fn = torch.nn.MSELoss()  # 定义去噪常用的 MSE 损失。
loss = loss_fn(denoised_images, clean_images)  # 计算去噪损失，输出 shape = ()。

print(\"Noisy image shape:\", noisy_images.shape)  # 打印输入图像 shape。
print(\"Denoised image shape:\", denoised_images.shape)  # 打印输出图像 shape。
print(\"Loss shape:\", loss.shape)  # 打印损失 shape。
