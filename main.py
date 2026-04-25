import torch  # 导入 PyTorch，用来定义 VAE 图像去噪模型；这里没有张量 shape。


class ConvVAE(torch.nn.Module):  # 定义一个最小卷积 VAE；输入是图像，输出是重建图像和隐变量统计量。
    def __init__(self, latent_dim: int = 16) -> None:  # 初始化 VAE 参数；latent_dim 是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.encoder = torch.nn.Sequential(  # 定义卷积 encoder；输入 shape = (B, 1, 28, 28)。
            torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # 第一层卷积后输出 shape = (B, 8, 14, 14)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 第二层卷积后输出 shape = (B, 16, 7, 7)。
            torch.nn.ReLU(),  # 再做激活，shape 不变。
        )  # 结束 encoder 定义；这里没有张量 shape。
        self.fc_mu = torch.nn.Linear(16 * 7 * 7, latent_dim)  # 定义均值层，输入 shape = (B, 784)，输出 shape = (B, latent_dim)。
        self.fc_logvar = torch.nn.Linear(16 * 7 * 7, latent_dim)  # 定义对数方差层，输入输出 shape 同上。
        self.fc_decode = torch.nn.Linear(latent_dim, 16 * 7 * 7)  # 定义解码前的线性层，输入 shape = (B, latent_dim)，输出 shape = (B, 784)。
        self.decoder = torch.nn.Sequential(  # 定义卷积 decoder；输入 shape = (B, 16, 7, 7)。
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 第一层反卷积后输出 shape = (B, 8, 14, 14)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 第二层反卷积后输出 shape = (B, 1, 28, 28)。
            torch.nn.Sigmoid(),  # 把像素范围压到 0 到 1，shape 不变。
        )  # 结束 decoder 定义；这里没有张量 shape。

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:  # 定义重参数化技巧；输入 mu 和 logvar 的 shape = (B, latent_dim)。
        std = torch.exp(0.5 * logvar)  # 把 logvar 变成标准差 std，shape = (B, latent_dim)。
        eps = torch.randn_like(std)  # 采样和 std 同 shape 的标准高斯噪声，shape = (B, latent_dim)。
        z = mu + eps * std  # 得到隐变量 z，shape = (B, latent_dim)。
        return z  # 返回采样后的隐变量。

    def forward(self, x: torch.Tensor):  # 定义前向传播；输入带噪图像 shape = (B, 1, 28, 28)。
        h = self.encoder(x)  # 输入 encoder，输出 shape = (B, 16, 7, 7)。
        h = h.flatten(1)  # 展平成二维矩阵，shape = (B, 784)。
        mu = self.fc_mu(h)  # 计算隐变量均值，shape = (B, latent_dim)。
        logvar = self.fc_logvar(h)  # 计算隐变量对数方差，shape = (B, latent_dim)。
        z = self.reparameterize(mu, logvar)  # 用重参数化技巧采样隐变量，shape = (B, latent_dim)。
        h_dec = self.fc_decode(z)  # 把隐变量映射回解码前特征，shape = (B, 784)。
        h_dec = h_dec.view(x.size(0), 16, 7, 7)  # 重新变回 4 维特征图，shape = (B, 16, 7, 7)。
        recon = self.decoder(h_dec)  # 输入 decoder 得到重建图像，shape = (B, 1, 28, 28)。
        return recon, mu, logvar  # 返回重建图像、均值和对数方差。


noisy_images = torch.rand(2, 1, 28, 28)  # 构造 2 张带噪图像，shape = (2, 1, 28, 28)。
clean_images = torch.rand(2, 1, 28, 28)  # 构造 2 张干净目标图像，shape = (2, 1, 28, 28)。
model = ConvVAE(latent_dim=16)  # 创建卷积 VAE 模型；模型本身没有 shape。
recon_images, mu, logvar = model(noisy_images)  # 前向传播得到重建图像和隐变量统计量；recon_images shape = (2, 1, 28, 28)。
recon_loss = torch.nn.functional.mse_loss(recon_images, clean_images)  # 计算重建损失，输出 shape = ()。
kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # 计算 KL 散度损失，输出 shape = ()。
loss = recon_loss + kl_loss  # 合并总损失，输出 shape = ()。

print(\"Noisy image shape:\", noisy_images.shape)  # 打印输入图像 shape。
print(\"Reconstructed image shape:\", recon_images.shape)  # 打印重建图像 shape。
print(\"Mu shape:\", mu.shape)  # 打印均值向量 shape。
print(\"Logvar shape:\", logvar.shape)  # 打印对数方差向量 shape。
print(\"Loss shape:\", loss.shape)  # 打印总损失 shape。
