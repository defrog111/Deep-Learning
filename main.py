import torch
import torch.nn as nn
import torch.optim as optim


print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
# =========================
# 1. 创建假数据
# y = 2x + 1
# =========================
X = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 2 * X + 1 + 0.2 * torch.rand(X.size())

# =========================
# 2. 定义模型
# =========================
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleNet()

# =========================
# 3. 定义 loss 和 optimizer
# =========================
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# =========================
# 4. 训练
# =========================
for epoch in range(100):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# =========================
# 5. 测试
# =========================
test_x = torch.tensor([[4.0]])
test_y = model(test_x)
print("Input:", test_x.item())
print("Predicted:", test_y.item())