import torch  # 导入 PyTorch，用来定义 Siamese 网络；这里没有张量 shape。


class Encoder(torch.nn.Module):  # 定义共享图像编码器；输入输出分别是图像张量和特征向量。
    def __init__(self) -> None:  # 初始化编码器参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.conv = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 定义卷积层，输入 shape = (B, 1, H, W)，输出 shape = (B, 8, H, W)。
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))  # 定义全局池化层，输出 shape = (B, 8, 1, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 1, 28, 28)。
        x = torch.relu(self.conv(x))  # 做卷积加激活，输出 shape = (B, 8, 28, 28)。
        x = self.pool(x)  # 做全局池化，输出 shape = (B, 8, 1, 1)。
        x = x.flatten(1)  # 展平成特征向量，shape = (B, 8)。
        return x  # 返回图像特征向量。


encoder = Encoder()  # 创建共享编码器；模型本身没有 shape。
image_a = torch.randn(2, 1, 28, 28)  # 构造第一路图像张量，shape = (2, 1, 28, 28)。
image_b = torch.randn(2, 1, 28, 28)  # 构造第二路图像张量，shape = (2, 1, 28, 28)。
feat_a = encoder(image_a)  # 提取第一路特征，shape = (2, 8)。
feat_b = encoder(image_b)  # 提取第二路特征，shape = (2, 8)。
distance = torch.norm(feat_a - feat_b, dim=1)  # 计算两路特征之间的欧式距离，shape = (2,)。

print(\"Feature A shape:\", feat_a.shape)  # 打印第一路特征 shape。
print(\"Feature B shape:\", feat_b.shape)  # 打印第二路特征 shape。
print(\"Distance shape:\", distance.shape)  # 打印距离向量 shape。

# å¸¸ç”¨ Metric å·²è¡¥å……ï¼Œä¸‹é¢æ˜¯è¿™é“é¢˜æœ€å¸¸è§æŒ‡æ ‡çš„æœ€å°ç¤ºä¾‹ã€‚
cosine_similarity = torch.nn.functional.cosine_similarity(feat_a, feat_b)  # è®¡ç®—ä¸¤è·¯ç‰¹å¾çš„ä½™å¼¦ç›¸ä¼¼åº¦ï¼Œshape = (2,)ã€‚
print("Cosine similarity:", cosine_similarity)  # æ‰“å°ä½™å¼¦ç›¸ä¼¼åº¦ã€‚
