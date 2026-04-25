import torch  # 导入 PyTorch，用来定义最小 GAN 的生成器和判别器；这里没有张量 shape。


class Generator(torch.nn.Module):  # 定义生成器；输入噪声向量，输出图像张量。
    def __init__(self, noise_dim: int = 16) -> None:  # 初始化生成器参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.fc = torch.nn.Linear(noise_dim, 28 * 28)  # 把噪声向量映射成图像像素向量，输入 shape = (B, noise_dim)，输出 shape = (B, 784)。

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 z 的 shape = (B, noise_dim)。
        x = torch.sigmoid(self.fc(z))  # 把噪声映射成 0 到 1 范围的像素向量，shape = (B, 784)。
        x = x.view(z.size(0), 1, 28, 28)  # 重排成图像张量，shape = (B, 1, 28, 28)。
        return x  # 返回生成图像。


class Discriminator(torch.nn.Module):  # 定义判别器；输入图像，输出真假分数。
    def __init__(self) -> None:  # 初始化判别器参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.fc = torch.nn.Linear(28 * 28, 1)  # 把图像像素向量映射成一个真假分数，输入 shape = (B, 784)，输出 shape = (B, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 1, 28, 28)。
        x = x.flatten(1)  # 展平成像素向量，shape = (B, 784)。
        score = self.fc(x)  # 输出真假 logits，shape = (B, 1)。
        return score  # 返回判别分数。


noise = torch.randn(2, 16)  # 构造两条噪声向量，shape = (2, 16)。
generator = Generator(noise_dim=16)  # 创建生成器；模型本身没有 shape。
discriminator = Discriminator()  # 创建判别器；模型本身没有 shape。
fake_images = generator(noise)  # 生成假图像，shape = (2, 1, 28, 28)。
fake_scores = discriminator(fake_images)  # 判别假图像，输出 shape = (2, 1)。

print(\"Noise shape:\", noise.shape)  # 打印噪声向量 shape。
print(\"Fake image shape:\", fake_images.shape)  # 打印生成图像 shape。
print(\"Fake score shape:\", fake_scores.shape)  # 打印判别器输出 shape。
