import torch  # 导入 PyTorch，用来定义 CNN、LoRA、量化和剪枝示例；这里没有张量 shape。
import torch.nn.utils.prune as prune  # 导入 PyTorch 的剪枝工具；这里没有张量 shape。


class LoRALinear(torch.nn.Module):  # 定义一个最小 LoRA 线性层；本质是原线性层加一个低秩增量。
    def __init__(self, in_features: int, out_features: int, rank: int = 2) -> None:  # 初始化 LoRA 参数；in_features、out_features 和 rank 都是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.base = torch.nn.Linear(in_features, out_features)  # 定义原始线性层，输入 shape = (B, in_features)，输出 shape = (B, out_features)。
        self.lora_a = torch.nn.Linear(in_features, rank, bias=False)  # 定义 LoRA 的 A 矩阵，输出 shape = (B, rank)。
        self.lora_b = torch.nn.Linear(rank, out_features, bias=False)  # 定义 LoRA 的 B 矩阵，输出 shape = (B, out_features)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape = (B, in_features)。
        base_out = self.base(x)  # 计算原始线性层输出，shape = (B, out_features)。
        lora_out = self.lora_b(self.lora_a(x))  # 计算低秩增量输出，shape = (B, out_features)。
        out = base_out + lora_out  # 把 LoRA 增量加到原输出上，shape = (B, out_features)。
        return out  # 返回最终输出。


class SimpleCNN(torch.nn.Module):  # 定义一个很简单的 CNN 分类模型；适合同时演示剪枝、量化和 LoRA。
    def __init__(self, num_classes: int = 3) -> None:  # 初始化模型参数；num_classes 是整数。
        super().__init__()  # 调用父类初始化；这里没有张量 shape。
        self.features = torch.nn.Sequential(  # 定义特征提取部分；输入是图像张量，输出是特征图张量。
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),  # 第一层卷积，输入 shape = (B, 1, 16, 16)，输出 shape = (B, 8, 16, 16)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.MaxPool2d(kernel_size=2),  # 下采样，输出 shape = (B, 8, 8, 8)。
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 第二层卷积，输出 shape = (B, 16, 8, 8)。
            torch.nn.ReLU(),  # 做激活，shape 不变。
            torch.nn.AdaptiveAvgPool2d((1, 1)),  # 做全局平均池化，输出 shape = (B, 16, 1, 1)。
        )  # 结束特征提取模块定义；这里没有张量 shape。
        self.classifier = LoRALinear(16, num_classes, rank=2)  # 定义带 LoRA 的分类头，输入 shape = (B, 16)，输出 shape = (B, num_classes)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入图像 shape = (B, 1, 16, 16)。
        x = self.features(x)  # 输入 CNN 特征提取层，输出 shape = (B, 16, 1, 1)。
        x = x.flatten(1)  # 展平成二维特征矩阵，shape = (B, 16)。
        logits = self.classifier(x)  # 输出分类 logits，shape = (B, num_classes)。
        return logits  # 返回分类结果。


torch.manual_seed(7)  # 固定随机种子，让这个最小训练例子的输出更稳定。
images = torch.randn(4, 1, 16, 16)  # 构造 4 张灰度图像，shape = (4, 1, 16, 16)。
labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)  # 构造分类标签向量，shape = (4,)。
model = SimpleCNN(num_classes=3)  # 创建简单 CNN 模型；模型本身没有 shape。
loss_fn = torch.nn.CrossEntropyLoss()  # 定义分类常用交叉熵损失；loss_fn 本身没有 shape。
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # 定义优化器，让 CNN 和 LoRA 分类头一起更新参数。

model.train()  # 切到训练模式，后面开始做最小训练循环。
for epoch in range(200):  # 训练 200 轮，让这个小模型先学会当前这 4 个样本。
    optimizer.zero_grad()  # 清空上一轮梯度。
    logits = model(images)  # 前向传播得到分类 logits，shape = (4, 3)。
    loss = loss_fn(logits, labels)  # 计算交叉熵损失，输出是标量张量，shape = ()。
    loss.backward()  # 反向传播，计算各参数梯度。
    optimizer.step()  # 根据梯度更新模型参数。

    if epoch % 50 == 0 or epoch == 199:  # 打印少量轮次，观察训练 loss 是否下降。
        print(f"Epoch {epoch:03d} | Train loss: {loss.item():.6f}")

model.eval()  # 切到推理模式，后面的指标计算和量化转换都按评估流程来做。
with torch.no_grad():  # 评估阶段不需要梯度。
    logits = model(images)  # 用训练后的模型重新做一次前向传播，shape = (4, 3)。
    loss = loss_fn(logits, labels)  # 计算评估损失，输出是标量张量，shape = ()。
    pred_labels = torch.argmax(logits, dim=1)  # 把 logits 转成预测类别，shape = (4,)。
    accuracy = (pred_labels == labels).float().mean()  # 计算 Accuracy，输出是标量。
    tp = ((pred_labels == 1) & (labels == 1)).float().sum()  # 计算以类别 1 为例的 TP，输出是标量。
    fp = ((pred_labels == 1) & (labels != 1)).float().sum()  # 计算以类别 1 为例的 FP，输出是标量。
    fn = ((pred_labels != 1) & (labels == 1)).float().sum()  # 计算以类别 1 为例的 FN，输出是标量。
    precision = tp / (tp + fp + 1e-7)  # 计算 Precision，输出是标量。
    recall = tp / (tp + fn + 1e-7)  # 计算 Recall，输出是标量。
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # 计算 F1，输出是标量。

prune.l1_unstructured(model.features[0], name="weight", amount=0.3)  # 训练完成后对第一层卷积权重做 30% 非结构化剪枝；这里主要演示 pruning 的用法。
conv_weight = model.features[0].weight_mask  # 取出剪枝后的 mask 张量，shape = (8, 1, 3, 3)。
sparsity = 1.0 - conv_weight.mean()  # 计算剪枝后的稀疏率，输出是标量。
float_classifier = model.classifier.base.eval()  # 取出训练后的分类头原始 Linear 层，并切到 eval 模式，便于做动态量化转换。
torch.backends.quantized.engine = "qnnpack"  # 当前环境支持 qnnpack；显式设置量化后端，避免出现 NoQEngine 错误。
float_classifier.qconfig = torch.ao.quantization.default_dynamic_qconfig  # 给浮点 Linear 层挂上动态量化配置，指定权重按 qint8 量化。
quantized_classifier = torch.nn.quantized.dynamic.Linear.from_float(float_classifier)  # 用未弃用的 from_float 接口把浮点 Linear 转成动态量化 Linear。
dummy_feature = torch.randn(4, 16)  # 构造一组假特征，shape = (4, 16)，用来测试量化后的分类头。
quant_logits = quantized_classifier(dummy_feature)  # 输入量化后的 Linear 层，输出 shape = (4, 3)。
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 统计当前模型可训练参数量，输出是整数。

print("Image shape:", images.shape)  # 打印输入图像 shape。
print("Logits shape:", logits.shape)  # 打印分类 logits shape。
print("Loss shape:", loss.shape)  # 打印损失张量 shape。
print("Accuracy:", float(accuracy))  # 打印 Accuracy。
print("Precision:", float(precision))  # 打印 Precision。
print("Recall:", float(recall))  # 打印 Recall。
print("F1:", float(f1))  # 打印 F1。
print("Sparsity:", float(sparsity))  # 打印剪枝后的稀疏率。
print("Quantized logits shape:", quant_logits.shape)  # 打印量化分类头输出 shape。
print("Trainable params:", int(trainable_params))  # 打印可训练参数量。



from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

img = Image.open("xxx.png")
x = transform(img)   # shape: (1, H, W)
x = x.unsqueeze(0)   # shape: (1, 1, H, W)


#NP
import numpy as np
arr = np.random.rand(16, 16)
x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
