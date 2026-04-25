import torch  # 导入 PyTorch，用来定义 ResNet 风格模型；这里没有张量 shape。


class ResidualBlock(torch.nn.Module):  # 定义最小残差块；输入输出都是 4 维图像特征张量。
    def __init__(self, channels: int) -> None:  # 初始化残差块；channels 是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第一个卷积层，输入输出 shape 都是 (B, channels, H, W)。
        self.relu = torch.nn.ReLU()  # 定义激活层，shape 不变。
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)  # 第二个卷积层，输入输出 shape 不变。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, channels, H, W)。
        identity = x  # 保存残差支路，shape = (B, channels, H, W)。
        out = self.conv1(x)  # 做第一层卷积，shape 不变。
        out = self.relu(out)  # 做激活，shape 不变。
        out = self.conv2(out)  # 做第二层卷积，shape 不变。
        out = out + identity  # 做残差相加，shape 不变。
        out = self.relu(out)  # 再做一次激活，shape 不变。
        return out  # 返回残差块输出。


class TinyResNet(torch.nn.Module):  # 定义一个最小 ResNet 分类模型。
    def __init__(self, num_classes: int = 10) -> None:  # 初始化网络参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.stem = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 定义输入 stem 卷积层，输出 shape = (B, 16, H, W)。
        self.block = ResidualBlock(16)  # 定义一个残差块，输入输出 shape 都是 (B, 16, H, W)。
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # 做全局平均池化，输出 shape = (B, 16, 1, 1)。
        self.head = torch.nn.Linear(16, num_classes)  # 定义分类头，输入 shape = (B, 16)，输出 shape = (B, num_classes)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 3, 32, 32)。
        x = self.stem(x)  # 做 stem 卷积，shape = (B, 16, 32, 32)。
        x = self.block(x)  # 输入残差块，shape 不变。
        x = self.pool(x)  # 做全局平均池化，shape = (B, 16, 1, 1)。
        x = x.flatten(1)  # 展平成二维矩阵，shape = (B, 16)。
        logits = self.head(x)  # 输出分类 logits，shape = (B, 10)。
        return logits  # 返回分类结果。


images = torch.randn(4, 3, 32, 32)  # 构造 4 张假图像，shape = (4, 3, 32, 32)。
model = TinyResNet()  # 创建 ResNet 模型；模型本身没有 shape。
logits = model(images)  # 前向传播得到输出，shape = (4, 10)。
print(\"Image shape:\", images.shape)  # 打印输入图像 shape。
print(\"Logits shape:\", logits.shape)  # 打印输出 logits shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
labels = torch.randint(0, 10, (4,))  # æž„é€ åˆ†ç±»æ ‡ç­¾å‘é‡ï¼Œshape = (4,)ã€‚
pred_labels = torch.argmax(logits, dim=1)  # æŠŠ logits è½¬æˆé¢„æµ‹ç±»åˆ«ï¼Œshape = (4,)ã€‚
accuracy = (pred_labels == labels).float().mean()  # è®¡ç®— Accuracyï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Accuracy:", float(accuracy))  # æ‰“å° Accuracyã€‚
