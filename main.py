import torch
import torch.nn as nn
import torch.optim as optim


class LogisticRegression(nn.Module):
    # 这是最小 PyTorch 二分类逻辑回归模型。
    # 公式是：
    # z = x @ W + b
    # p = sigmoid(z)
    # 这里 forward 直接返回概率 p，所以 loss 选 BCELoss。
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 输入 1 维特征，输出 1 个 logit；weight shape = (1, 1)，bias shape = (1,)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的 shape = (batch_size, 1)。
        logits = self.linear(x)  # logits shape = (batch_size, 1)。
        prob = torch.sigmoid(logits)  # sigmoid 后得到概率，shape 仍然是 (batch_size, 1)。
        return prob


def main() -> None:
    # 1. 构造最小二分类数据；X shape = (4, 1)，y shape = (4, 1)。
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)
    y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32)

    # 2. 定义模型。
    model = LogisticRegression()

    # 3. 定义损失函数。
    # 这里因为 forward 已经手动加了 sigmoid，所以配 BCELoss。
    criterion = nn.BCELoss()

    # 4. 定义优化器。
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 5. 训练。
    model.train()
    for epoch in range(1000):
        pred = model(X)  # pred shape = (4, 1)，这里返回的是概率，不是原始 logit。
        loss = criterion(pred, y)  # BCE loss 输出是标量张量，shape = ()。

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == 999:
            print(f"Epoch {epoch:04d} | Loss {loss.item():.6f}")

    # 6. 测试 / inference。
    model.eval()
    with torch.no_grad():
        test = torch.tensor([[1.5]], dtype=torch.float32)  # 单条测试输入，shape = (1, 1)。
        prob = model(test)  # prob shape = (1, 1)。
        pred_label = (prob >= 0.5).float()  # threshold = 0.5，把概率转成预测类别。
        print("\nTest Input Shape:", tuple(test.shape))
        print("Predicted Probability:", prob.item())
        print("Predicted Label:", pred_label.item())

    # 顺手提醒你：
    # 1. 这版能跑，但数值稳定性不如 BCEWithLogitsLoss 版。
    # 2. 更稳的写法通常是不在 forward 里手动 sigmoid，而是把 raw logits 直接交给 BCEWithLogitsLoss。


if __name__ == "__main__":
    main()
