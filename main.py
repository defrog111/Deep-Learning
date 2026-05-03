# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
"""Core interview problem: PyTorch knowledge distillation with KLDivLoss."""  # 说明这道题是蒸馏基础题。

import torch  # 导入 PyTorch，用来定义模型和训练流程。


torch.manual_seed(7)  # 固定随机种子，保证每次运行结果更稳定。


class TeacherClassifier(torch.nn.Module):  # 定义 teacher 模型，容量更大一些。
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(  # 用两层隐藏层构造一个稍大的 MLP teacher。
            torch.nn.Linear(input_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 返回 teacher logits，shape = (B, C)。


class StudentClassifier(torch.nn.Module):  # 定义 student 模型，容量更小一些。
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(  # 只保留一层较小隐藏层，方便体现蒸馏意义。
            torch.nn.Linear(input_dim, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 返回 student logits，shape = (B, C)。


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:  # 计算分类准确率。
    predictions = torch.argmax(logits, dim=1)  # 取每行最大 logit 对应的类别索引，shape = (B,)。
    return (predictions == labels).float().mean().item()  # 返回准确率标量。


x = torch.tensor(  # 构造一个简单的三分类二维特征数据集，shape = (12, 2)。
    [
        [-2.0, -1.0],
        [-1.8, -1.3],
        [-1.6, -0.8],
        [-1.2, -1.4],
        [1.1, 1.8],
        [1.5, 1.2],
        [1.7, 1.9],
        [1.0, 1.1],
        [2.2, -1.6],
        [2.6, -1.2],
        [2.0, -0.8],
        [2.5, -1.9],
    ],
    dtype=torch.float32,
)
labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)  # 构造真实类别标签，shape = (12,)。

teacher = TeacherClassifier(input_dim=2, num_classes=3)  # 实例化 teacher 模型。
student = StudentClassifier(input_dim=2, num_classes=3)  # 实例化 student 模型。

ce_loss_fn = torch.nn.CrossEntropyLoss()  # 定义 hard label 的交叉熵损失。
kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")  # 定义蒸馏用的 KL 散度损失。

teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.03)  # 定义 teacher 优化器。
student_optimizer = torch.optim.Adam(student.parameters(), lr=0.03)  # 定义 student 优化器。

temperature = 2.0  # 定义蒸馏温度，用来把 teacher 和 student 分布变软。
alpha = 0.7  # 定义软目标损失权重。
  
teacher.train()  # 切到 teacher 训练模式。
for epoch in range(300):  # 先单独训练 teacher。
    teacher_optimizer.zero_grad()  # 清空上一轮 teacher 梯度。
    teacher_logits = teacher(x)  # 前向传播得到 teacher logits，shape = (12, 3)。
    teacher_loss = ce_loss_fn(teacher_logits, labels)  # 用真实标签训练 teacher。
    teacher_loss.backward()  # 反向传播 teacher 梯度。
    teacher_optimizer.step()  # 更新 teacher 参数。

teacher.eval()  # 切到 teacher 推理模式，后面只用它提供软标签。
with torch.no_grad():
    teacher_logits = teacher(x)  # 固定 teacher 输出，shape = (12, 3)。
    teacher_probs = torch.softmax(teacher_logits, dim=1)  # 观察 teacher 普通概率分布，shape = (12, 3)。
    teacher_accuracy = accuracy_from_logits(teacher_logits, labels)  # 计算 teacher 训练后的准确率。

student.train()  # 切到 student 训练模式。
for epoch in range(400):  # 用 hard label + soft target 一起训练 student。
    student_optimizer.zero_grad()  # 清空上一轮 student 梯度。
    student_logits = student(x)  # 前向传播得到 student logits，shape = (12, 3)。

    hard_loss = ce_loss_fn(student_logits, labels)  # 计算 student 对真实标签的交叉熵损失。
    soft_student = torch.log_softmax(student_logits / temperature, dim=1)  # 计算 student 的软化 log prob，shape = (12, 3)。
    soft_teacher = torch.softmax(teacher_logits / temperature, dim=1)  # 计算 teacher 的软化 prob，shape = (12, 3)。
    distill_loss = kl_loss_fn(soft_student, soft_teacher) * (temperature ** 2)  # 计算 KL 蒸馏损失，并按温度平方做标准缩放。
    student_loss = alpha * distill_loss + (1.0 - alpha) * hard_loss  # 融合软目标和硬标签损失。

    student_loss.backward()  # 反向传播 student 梯度。
    student_optimizer.step()  # 更新 student 参数。

student.eval()  # 切到 student 推理模式。
with torch.no_grad():
    student_logits = student(x)  # 再跑一遍 student logits，shape = (12, 3)。
    student_probs = torch.softmax(student_logits, dim=1)  # 计算 student 普通概率分布，shape = (12, 3)。
    student_accuracy = accuracy_from_logits(student_logits, labels)  # 计算 student 最终准确率。

single_x = torch.tensor([[1.4, 1.6]], dtype=torch.float32)  # 构造单条推理样本，shape = (1, 2)。
with torch.no_grad():
    single_teacher_logits = teacher(single_x)  # 计算单条 teacher logits，shape = (1, 3)。
    single_student_logits = student(single_x)  # 计算单条 student logits，shape = (1, 3)。
    single_student_probs = torch.softmax(single_student_logits, dim=1)  # 计算单条 student 概率，shape = (1, 3)。
    single_prediction = torch.argmax(single_student_logits, dim=1)  # 取单条 student 预测类别，shape = (1,)。

print("Input shape:", x.shape)  # 打印输入特征张量 shape。
print("Labels shape:", labels.shape)  # 打印标签张量 shape。
print("Teacher logits shape:", teacher_logits.shape)  # 打印 teacher logits shape。
print("Teacher probs shape:", teacher_probs.shape)  # 打印 teacher 概率 shape。
print("Student logits shape:", student_logits.shape)  # 打印 student logits shape。
print("Student probs shape:", student_probs.shape)  # 打印 student 概率 shape。
print("Teacher CE loss shape:", teacher_loss.shape)  # 打印 teacher loss 的 shape。
print("Hard loss shape:", hard_loss.shape)  # 打印 hard loss 的 shape。
print("Distill loss shape:", distill_loss.shape)  # 打印蒸馏损失的 shape。
print("Student total loss shape:", student_loss.shape)  # 打印 student 总损失的 shape。
print("Teacher accuracy:", round(teacher_accuracy, 4))  # 打印 teacher 最终准确率。
print("Student accuracy:", round(student_accuracy, 4))  # 打印 student 最终准确率。
print("Single input shape:", single_x.shape)  # 打印单条推理输入 shape。
print("Single teacher logits shape:", single_teacher_logits.shape)  # 打印单条 teacher logits shape。
print("Single student logits shape:", single_student_logits.shape)  # 打印单条 student logits shape。
print("Single student probs:", single_student_probs)  # 打印单条 student 概率。
print("Single student prediction:", single_prediction)  # 打印单条 student 预测类别。
