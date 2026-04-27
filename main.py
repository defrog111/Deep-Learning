import torch
import torch.nn as nn
import torch.optim as optim


class LogisticRegression(nn.Module):
    # 这是更稳定的 PyTorch 二分类逻辑回归版本。
    # 核心思路：
    # z = x @ W + b
    # 训练时直接把 z（raw logits）交给 BCEWithLogitsLoss
    # 推理时再手动 sigmoid，把 logits 变成概率
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入 1 维特征，输出 1 个 raw logit；weight shape = (1, 1)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的 shape = (batch_size, 1)。
        logits = self.linear(x)  # logits shape = (batch_size, 1)；这里故意不加 sigmoid。
        return logits


def main() -> None:
    # 1. 构造最小二分类数据；X shape = (4, 1)，y shape = (4, 1)。
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)

    # 2. 定义模型。
    model = LogisticRegression()

    # 3. 定义更稳定的损失函数。
    # BCEWithLogitsLoss 内部已经把 sigmoid 和 BCE 合在一起做了，所以数值更稳定。
    criterion = nn.BCEWithLogitsLoss()

    # 4. 定义优化器。
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 5. 训练。
    model.train()
    for epoch in range(1000):
        logits = model(X)  # logits shape = (4, 1)，这里是原始分数，不是概率。
        loss = criterion(logits, y)  # loss 输出是标量张量，shape = ()。

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 999:
            print(f"Epoch {epoch:04d} | Loss {loss.item():.6f}")

    # 6. 测试 / inference。
    model.eval()
    with torch.no_grad():
        test = torch.tensor([[1.5]], dtype=torch.float32)  # 单条测试输入，shape = (1, 1)。
        logits = model(test)  # logits shape = (1, 1)。
        prob = torch.sigmoid(logits)  # 推理时再手动 sigmoid，把 raw logits 变成概率。
        pred_label = (prob >= 0.5).float()
        print("\nTest Input Shape:", tuple(test.shape))
        print("Raw Logit:", logits.item())
        print("Predicted Probability:", prob.item())
        print("Predicted Label:", pred_label.item())

    # 顺手提醒你：
    # 1. 二分类里，这版通常比 “forward 里手动 sigmoid + BCELoss” 更推荐。
    # 2. 以后你看到多分类时，更常见对应关系是：raw logits + CrossEntropyLoss。


if __name__ == "__main__":
    main()
