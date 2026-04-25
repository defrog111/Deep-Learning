import torch  # 导入 PyTorch，用来定义模型和张量；这里没有张量 shape。


class DoubleConv(torch.nn.Module):  # 定义 U-Net 里常见的双卷积模块；输入输出通常都是 4 维图像张量。
    def __init__(self, in_channels: int, out_channels: int) -> None:  # 初始化卷积模块；通道数是整数，没有 shape。
        super().__init__()  # 调用父类初始化函数；这里没有张量 shape。
        self.block = torch.nn.Sequential(  # 用 Sequential 串起两个卷积层；整体输入输出都是 4 维张量。
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 第一个卷积层，输入 shape = (B, in_channels, H, W)，输出 shape = (B, out_channels, H, W)。
            torch.nn.ReLU(),  # 对卷积输出做激活，shape 不变。
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # 第二个卷积层，输入输出 shape 都是 (B, out_channels, H, W)。
            torch.nn.ReLU(),  # 再做一次激活，shape 不变。
        )  # 结束双卷积模块定义；这里没有张量 shape。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, C, H, W)。
        return self.block(x)  # 返回双卷积后的结果；输出 shape = (B, out_channels, H, W)。


class UNet(torch.nn.Module):  # 定义一个最小 U-Net；输入是图像，输出是像素级别的 logits 图。
    def __init__(self, in_channels: int = 3, num_classes: int = 2) -> None:  # 初始化 U-Net；通道数和类别数都是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.down1 = DoubleConv(in_channels, 16)  # 第一层编码器，输入 shape = (B, 3, H, W)，输出 shape = (B, 16, H, W)。
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)  # 下采样层，会把高宽减半。
        self.down2 = DoubleConv(16, 32)  # 第二层编码器，输入 shape = (B, 16, H/2, W/2)，输出 shape = (B, 32, H/2, W/2)。
        self.up = torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # 反卷积上采样层，把高宽放大回去。
        self.dec = DoubleConv(32, 16)  # 解码器卷积层，拼接 skip connection 后输入 shape = (B, 32, H, W)。
        self.head = torch.nn.Conv2d(16, num_classes, kernel_size=1)  # 最后的 1x1 卷积，把通道映射成类别数；输出 shape = (B, num_classes, H, W)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义 U-Net 前向传播；输入 shape = (B, 3, H, W)。
        x1 = self.down1(x)  # 第一层编码特征，shape = (B, 16, H, W)。
        x2 = self.pool1(x1)  # 下采样后的特征，shape = (B, 16, H/2, W/2)。
        x3 = self.down2(x2)  # 第二层编码特征，shape = (B, 32, H/2, W/2)。
        x4 = self.up(x3)  # 上采样后的特征，shape = (B, 16, H, W)。
        x5 = torch.cat([x4, x1], dim=1)  # 把上采样特征和 skip connection 按通道拼接，shape = (B, 32, H, W)。
        x6 = self.dec(x5)  # 解码后的特征，shape = (B, 16, H, W)。
        logits = self.head(x6)  # 输出像素分类 logits 图，shape = (B, num_classes, H, W)。
        return logits  # 返回分割 logits 图。


images = torch.randn(2, 3, 64, 64)  # 构造两张假图像，shape = (2, 3, 64, 64)。
masks = torch.randint(0, 2, (2, 64, 64))  # 构造两张像素标签图，shape = (2, 64, 64)。
model = UNet(in_channels=3, num_classes=2)  # 创建 U-Net 模型；模型本身不是张量，没有 shape。
logits = model(images)  # 前向传播得到输出，shape = (2, 2, 64, 64)。
loss_fn = torch.nn.CrossEntropyLoss()  # 定义分割常用的交叉熵损失；loss_fn 本身没有 shape。
loss = loss_fn(logits, masks)  # 计算分割损失，输出是标量张量，shape = ()。

print(\"Image shape:\", images.shape)  # 打印输入图像 shape。
print(\"Logits shape:\", logits.shape)  # 打印输出 logits 图的 shape。
print(\"Mask shape:\", masks.shape)  # 打印标签 mask 的 shape。
print(\"Loss shape:\", loss.shape)  # 打印损失张量的 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
pred_mask = torch.argmax(logits, dim=1)  # æŠŠåƒç´ åˆ†ç±» logits è½¬æˆé¢„æµ‹ maskï¼Œshape = (2, 64, 64)ã€‚
intersection = ((pred_mask == 1) & (masks == 1)).float().sum()  # è®¡ç®—å‰æ™¯ç±»åˆ«äº¤é›†åƒç´ æ•°ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
union = ((pred_mask == 1) | (masks == 1)).float().sum()  # è®¡ç®—å‰æ™¯ç±»åˆ«å¹¶é›†åƒç´ æ•°ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
iou = intersection / (union + 1e-7)  # è®¡ç®— IoUï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
dice = 2.0 * intersection / ((pred_mask == 1).float().sum() + (masks == 1).float().sum() + 1e-7)  # è®¡ç®— Diceï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
pixel_accuracy = (pred_mask == masks).float().mean()  # è®¡ç®—åƒç´ å‡†ç¡®çŽ‡ï¼Œè¾“å‡ºæ˜¯æ ‡é‡ã€‚
print("Pixel Accuracy:", float(pixel_accuracy))  # æ‰“å°åƒç´ å‡†ç¡®çŽ‡ã€‚
print("IoU:", float(iou))  # æ‰“å° IoUã€‚
print("Dice:", float(dice))  # æ‰“å° Diceã€‚
