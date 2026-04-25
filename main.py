import torch  # 导入 PyTorch，用来定义多标签 CNN 分类模型和张量；这里没有张量 shape。


class MultiLabelCNN(torch.nn.Module):  # 定义一个简单 CNN 多标签分类模型；输入是图像，输出是多标签 logits。
    def __init__(self, num_labels: int = 4) -> None:  # 初始化模型参数；num_labels 是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.features = torch.nn.Sequential(  # 定义特征提取层；输入是图像张量，输出是特征图张量。
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 第一层卷积，输入 shape = (B, 1, 16, 16)，输出 shape = (B, 8, 16, 16)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.MaxPool2d(kernel_size=2),  # 下采样，输出 shape = (B, 8, 8, 8)。
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 第二层卷积，输出 shape = (B, 16, 8, 8)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # 做全局池化，输出 shape = (B, 16, 1, 1)。
        )  # 结束特征提取层定义；这里没有张量 shape。
        self.head = torch.nn.Linear(16, num_labels)  # 定义多标签分类头，输入 shape = (B, 16)，输出 shape = (B, num_labels)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 1, 16, 16)。
        x = self.features(x)  # 输入 CNN 特征提取层，输出 shape = (B, 16, 1, 1)。
        x = x.flatten(1)  # 展平成二维特征矩阵，shape = (B, 16)。
        logits = self.head(x)  # 输出多标签 logits，shape = (B, num_labels)。
        return logits  # 返回多标签分类结果。


images = torch.randn(4, 1, 16, 16)  # 构造 4 张灰度图像，shape = (4, 1, 16, 16)。
targets = torch.tensor(  # 构造 4 条多标签目标，shape = (4, 4)。
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ],
    dtype=torch.float32,
)  # 结束多标签目标定义；整体 shape = (4, 4)。
model = MultiLabelCNN(num_labels=4)  # 创建多标签 CNN 模型；模型本身没有 shape。
logits = model(images)  # 前向传播得到多标签 logits，shape = (4, 4)。
loss_fn = torch.nn.BCEWithLogitsLoss()  # 定义多标签常用损失函数；loss_fn 本身没有 shape。
loss = loss_fn(logits, targets)  # 计算多标签损失，输出是标量张量，shape = ()。
probs = torch.sigmoid(logits)  # 把 logits 转成多标签概率，shape = (4, 4)。
preds = (probs >= 0.5).float()  # 按 0.5 阈值转成多标签预测，shape = (4, 4)。
tp = ((preds == 1) & (targets == 1)).float().sum()  # 计算所有标签的 TP，总和输出是标量。
fp = ((preds == 1) & (targets == 0)).float().sum()  # 计算所有标签的 FP，总和输出是标量。
fn = ((preds == 0) & (targets == 1)).float().sum()  # 计算所有标签的 FN，总和输出是标量。
precision = tp / (tp + fp + 1e-7)  # 计算 micro Precision，输出是标量。
recall = tp / (tp + fn + 1e-7)  # 计算 micro Recall，输出是标量。
f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # 计算 micro F1，输出是标量。
hamming_loss = (preds != targets).float().mean()  # 计算 Hamming Loss，输出是标量。
exact_match = (preds == targets).all(dim=1).float().mean()  # 计算 Exact Match Ratio，输出是标量。

print("Image shape:", images.shape)  # 打印输入图像 shape。
print("Target shape:", targets.shape)  # 打印多标签目标 shape。
print("Logits shape:", logits.shape)  # 打印多标签 logits shape。
print("Loss shape:", loss.shape)  # 打印损失张量 shape。
print("Micro Precision:", float(precision))  # 打印 micro Precision。
print("Micro Recall:", float(recall))  # 打印 micro Recall。
print("Micro F1:", float(f1))  # 打印 micro F1。
print("Hamming Loss:", float(hamming_loss))  # 打印 Hamming Loss。
print("Exact Match Ratio:", float(exact_match))  # 打印 Exact Match Ratio。
