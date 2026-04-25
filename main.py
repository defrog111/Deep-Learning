import torch  # 导入 PyTorch，用来定义 CNN 和 finetuning 示例；这里没有张量 shape。


class LoRALinear(torch.nn.Module):  # 定义一个最小 LoRA 线性层；方便演示 LoRA finetuning。
    def __init__(self, in_features: int, out_features: int, rank: int = 2) -> None:  # 初始化 LoRA 参数；维度和秩都是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.base = torch.nn.Linear(in_features, out_features)  # 定义原始线性层，输入 shape = (B, in_features)，输出 shape = (B, out_features)。
        self.lora_a = torch.nn.Linear(in_features, rank, bias=False)  # 定义 LoRA A 矩阵，输出 shape = (B, rank)。
        self.lora_b = torch.nn.Linear(rank, out_features, bias=False)  # 定义 LoRA B 矩阵，输出 shape = (B, out_features)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, in_features)。
        out = self.base(x) + self.lora_b(self.lora_a(x))  # 返回原输出加 LoRA 增量，shape = (B, out_features)。
        return out  # 返回最终输出。


class TinyBackbone(torch.nn.Module):  # 定义一个很小的 CNN backbone；适合演示不同 finetuning 策略。
    def __init__(self) -> None:  # 初始化 backbone 参数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.block1 = torch.nn.Sequential(  # 定义第一段卷积块；输入图像，输出特征图。
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 卷积后输出 shape = (B, 8, 16, 16)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.MaxPool2d(2),  # 下采样，输出 shape = (B, 8, 8, 8)。
        )  # 结束第一段卷积块定义；这里没有张量 shape。
        self.block2 = torch.nn.Sequential(  # 定义第二段卷积块；输入特征图，输出更深特征图。
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 卷积后输出 shape = (B, 16, 8, 8)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # 做全局平均池化，输出 shape = (B, 16, 1, 1)。
        )  # 结束第二段卷积块定义；这里没有张量 shape。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 1, 16, 16)。
        x = self.block1(x)  # 输入第一段卷积块，输出 shape = (B, 8, 8, 8)。
        x = self.block2(x)  # 输入第二段卷积块，输出 shape = (B, 16, 1, 1)。
        x = x.flatten(1)  # 展平成二维特征矩阵，shape = (B, 16)。
        return x  # 返回 backbone 特征。


class FinetuneCNN(torch.nn.Module):  # 定义一个最小 CNN 分类模型；用来演示不同 finetuning 策略。
    def __init__(self, num_classes: int = 3, use_lora: bool = False) -> None:  # 初始化模型参数；num_classes 是整数，use_lora 是布尔值。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.backbone = TinyBackbone()  # 创建 CNN backbone；输出特征 shape = (B, 16)。
        self.head = LoRALinear(16, num_classes) if use_lora else torch.nn.Linear(16, num_classes)  # 根据需求创建普通 head 或 LoRA head。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 1, 16, 16)。
        feat = self.backbone(x)  # 提取 backbone 特征，shape = (B, 16)。
        logits = self.head(feat)  # 输出分类 logits，shape = (B, num_classes)。
        return logits  # 返回分类结果。


images = torch.randn(4, 1, 16, 16)  # 构造 4 张灰度图像，shape = (4, 1, 16, 16)。
labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)  # 构造分类标签向量，shape = (4,)。
model = FinetuneCNN(num_classes=3, use_lora=True)  # 创建带 LoRA head 的 CNN；模型本身没有 shape。

for param in model.backbone.parameters():  # 遍历 backbone 所有参数。
    param.requires_grad = False  # 这一步对应 linear probing：先冻结整个 backbone，只训练 head。

for param in model.backbone.block2.parameters():  # 遍历 backbone 的最后一个卷积块参数。
    param.requires_grad = True  # 这一步对应 partial unfreeze：解冻最后一层或最后一个 block。

optimizer = torch.optim.Adam(  # 创建带 differential learning rate 的优化器；不同参数组用不同学习率。
    [
        {"params": model.backbone.block2.parameters(), "lr": 1e-4},  # 给解冻的 backbone 最后一块用更小学习率。
        {"params": model.head.parameters(), "lr": 1e-3},  # 给分类头用更大学习率。
    ]
)  # 结束优化器定义；这里没有张量 shape。

logits = model(images)  # 前向传播得到分类 logits，shape = (4, 3)。
loss_fn = torch.nn.CrossEntropyLoss()  # 定义分类交叉熵损失；loss_fn 本身没有 shape。
loss = loss_fn(logits, labels)  # 计算分类损失，输出是标量张量，shape = ()。
pred_labels = torch.argmax(logits, dim=1)  # 把 logits 转成预测类别，shape = (4,)。
accuracy = (pred_labels == labels).float().mean()  # 计算 Accuracy，输出是标量。
tp = ((pred_labels == 1) & (labels == 1)).float().sum()  # 计算以类别 1 为例的 TP，输出是标量。
fp = ((pred_labels == 1) & (labels != 1)).float().sum()  # 计算以类别 1 为例的 FP，输出是标量。
fn = ((pred_labels != 1) & (labels == 1)).float().sum()  # 计算以类别 1 为例的 FN，输出是标量。
precision = tp / (tp + fp + 1e-7)  # 计算 Precision，输出是标量。
recall = tp / (tp + fn + 1e-7)  # 计算 Recall，输出是标量。
f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # 计算 F1，输出是标量。
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计当前可训练参数量，输出是整数。

print("Image shape:", images.shape)  # 打印输入图像 shape。
print("Label shape:", labels.shape)  # 打印标签 shape。
print("Logits shape:", logits.shape)  # 打印分类 logits shape。
print("Loss shape:", loss.shape)  # 打印损失张量 shape。
print("Accuracy:", float(accuracy))  # 打印 Accuracy。
print("Precision:", float(precision))  # 打印 Precision。
print("Recall:", float(recall))  # 打印 Recall。
print("F1:", float(f1))  # 打印 F1。
print("Trainable params:", int(trainable_params))  # 打印当前可训练参数量。
