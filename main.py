import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot，用来画 loss 和 R2 曲线；这里没有张量 shape。
import torch  # 导入 PyTorch，用来创建张量、模型和训练流程；这里没有张量 shape。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集读取函数；这里没有张量 shape。
from sklearn.model_selection import train_test_split  # 导入数据切分函数；这里没有张量 shape。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具；这里没有张量 shape。
from torch.utils.data import DataLoader, TensorDataset  # 导入小批量数据工具；这里没有张量 shape。


class Net(torch.nn.Module):  # 定义一个线性回归网络类；输入 x 的常见 shape 是 (batch_size, 8)，输出常见 shape 是 (batch_size, 1)。
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:  # 初始化网络；input_dim 这里会是 8，output_dim 这里会是 1。
        super().__init__()  # 调用父类构造函数，让当前类具备 Module 能力；这里没有张量 shape。
        self.linear = torch.nn.Linear(input_dim, output_dim)  # 定义线性层；权重 shape 是 (1, 8)，偏置 shape 是 (1,)。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播；输入 x 的 shape 常见是 (batch_size, 8) 或 (N, 8)。
        return self.linear(x)  # 返回线性层输出；输出 shape 是 (batch_size, 1) 或 (N, 1)。


if __name__ == "__main__":  # 只有直接运行 main.py 时才执行下面代码；这里没有张量 shape。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择运行设备；device 是一个设备对象，不是张量，没有 shape。
    print(f"Using device: {device}")  # 打印当前使用的设备；这里没有张量 shape。

    dataset = fetch_california_housing()  # 读取加州房价数据集对象；dataset 不是张量，没有 shape。
    features = dataset.data  # 取出特征矩阵；features 的 shape 是 (20640, 8)。
    targets = dataset.target  # 取出回归标签；targets 的 shape 是 (20640,)。

    x_train_full, x_test, y_train_full, y_test = train_test_split(  # 先切出训练+验证集合和测试集合。
        features,  # 传入全部特征；shape 是 (20640, 8)。
        targets,  # 传入全部标签；shape 是 (20640,)。
        test_size=0.2,  # 20% 数据给测试集；这不是张量，没有 shape。
        random_state=42,  # 固定随机种子保证可复现；这不是张量，没有 shape。
    )  # 切分后 x_train_full shape 是 (16512, 8)，x_test shape 是 (4128, 8)，y_train_full shape 是 (16512,)，y_test shape 是 (4128,)。
    x_train, x_val, y_train, y_val = train_test_split(  # 再从训练+验证集合中切出真正的训练集和验证集。
        x_train_full,  # 传入训练+验证特征；shape 是 (16512, 8)。
        y_train_full,  # 传入训练+验证标签；shape 是 (16512,)。
        test_size=0.2,  # 再切出 20% 作为验证集；这不是张量，没有 shape。
        random_state=42,  # 固定随机种子；这不是张量，没有 shape。
    )  # 切分后 x_train shape 大约是 (13209, 8)，x_val shape 大约是 (3303, 8)，y_train shape 大约是 (13209,)，y_val shape 大约是 (3303,)。

    scaler = StandardScaler()  # 创建标准化器对象；scaler 不是张量，没有 shape。
    x_train = scaler.fit_transform(x_train)  # 用训练集拟合并标准化训练特征；x_train 的 shape 仍然是 (13209, 8)。
    x_val = scaler.transform(x_val)  # 用训练集统计量标准化验证特征；x_val 的 shape 仍然是 (3303, 8)。
    x_test = scaler.transform(x_test)  # 用训练集统计量标准化测试特征；x_test 的 shape 仍然是 (4128, 8)。

    x_train = torch.tensor(x_train, dtype=torch.float32)  # 把训练特征转成 float32 张量；x_train 的 shape 是 (13209, 8)。
    y_train = torch.tensor(y_train, dtype=torch.float32)  # 把训练标签转成 float32 张量；y_train 的 shape 是 (13209,)。
    x_val = torch.tensor(x_val, dtype=torch.float32)  # 把验证特征转成 float32 张量；x_val 的 shape 是 (3303, 8)。
    y_val = torch.tensor(y_val, dtype=torch.float32)  # 把验证标签转成 float32 张量；y_val 的 shape 是 (3303,)。
    x_test = torch.tensor(x_test, dtype=torch.float32)  # 把测试特征转成 float32 张量；x_test 的 shape 是 (4128, 8)。
    y_test = torch.tensor(y_test, dtype=torch.float32)  # 把测试标签转成 float32 张量；y_test 的 shape 是 (4128,)。

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)  # 创建训练数据加载器；每个 batch_x shape 常见是 (32, 8)，每个 batch_y shape 常见是 (32,)。

    x_train_eval = x_train.to(device)  # 复制一份训练特征到 device 上给整集评估使用；x_train_eval 的 shape 是 (13209, 8)。
    y_train_eval = y_train.to(device)  # 复制一份训练标签到 device 上给整集评估使用；y_train_eval 的 shape 是 (13209,)。
    x_val = x_val.to(device)  # 把验证特征放到 device 上；x_val 的 shape 是 (3303, 8)。
    y_val = y_val.to(device)  # 把验证标签放到 device 上；y_val 的 shape 是 (3303,)。
    x_test = x_test.to(device)  # 把测试特征放到 device 上；x_test 的 shape 是 (4128, 8)。
    y_test = y_test.to(device)  # 把测试标签放到 device 上；y_test 的 shape 是 (4128,)。

    net = Net(input_dim=x_train.shape[1], output_dim=1).to(device)  # 创建网络并放到 device 上；net(batch_x) 的输出 shape 会是 (batch_size, 1)。
    loss_fn = torch.nn.MSELoss()  # 定义均方误差损失函数；loss_fn 本身不是张量，没有 shape。
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 定义 SGD 优化器；optimizer 不是张量，没有 shape。
    epochs = 300  # 设定训练轮数；这是整数，没有 shape。

    train_loss_history = []  # 记录每一轮训练集 loss；列表长度最终会是 300。
    val_loss_history = []  # 记录每一轮验证集 loss；列表长度最终会是 300。
    train_r2_history = []  # 记录每一轮训练集 R2；列表长度最终会是 300。
    val_r2_history = []  # 记录每一轮验证集 R2；列表长度最终会是 300。

    for epoch in range(epochs):  # 开始训练循环；epoch 是整数，没有 shape。
        net.train()  # 把网络切到训练模式；这里没有张量 shape。
        total_train_loss = 0.0  # 记录当前 epoch 所有 batch 的 loss 总和；这是浮点数，没有 shape。

        for batch_x, batch_y in train_loader:  # 从 DataLoader 中逐批取数据；batch_x shape 常见是 (32, 8)，batch_y shape 常见是 (32,)。
            batch_x = batch_x.to(device)  # 把当前 batch 特征搬到 device 上；batch_x 的 shape 仍然是 (32, 8)。
            batch_y = batch_y.to(device)  # 把当前 batch 标签搬到 device 上；batch_y 的 shape 仍然是 (32,)。
            predictions = net(batch_x).squeeze(1)  # 前向传播并压掉最后一维；squeeze 前 shape 是 (32, 1)，squeeze 后 shape 是 (32,)。
            loss = loss_fn(predictions, batch_y)  # 计算当前 batch 的 MSE loss；loss 是 0 维标量张量，shape 是 ()。

            optimizer.zero_grad()  # 清空上一轮反向传播留下的梯度；这里没有张量 shape。
            loss.backward()  # 对 loss 做反向传播；这里没有直接新建可见张量 shape。
            optimizer.step()  # 按梯度更新参数；这里没有张量 shape。

            total_train_loss += loss.item()  # 把当前 batch 的标量 loss 累加起来；结果是浮点数，没有 shape。

        avg_train_loss = total_train_loss / len(train_loader)  # 计算当前 epoch 的平均训练 loss；结果是浮点数，没有 shape。

        net.eval()  # 把网络切到评估模式；这里没有张量 shape。
        with torch.no_grad():  # 关闭梯度计算，评估时更省显存；这里没有张量 shape。
            train_predictions = net(x_train_eval).squeeze(1)  # 对整个训练集做预测；squeeze 前 shape 是 (13209, 1)，squeeze 后 shape 是 (13209,)。
            val_predictions = net(x_val).squeeze(1)  # 对整个验证集做预测；squeeze 前 shape 是 (3303, 1)，squeeze 后 shape 是 (3303,)。

            train_loss = loss_fn(train_predictions, y_train_eval).item()  # 计算整集训练 loss；train_loss 是浮点数，没有 shape。
            val_loss = loss_fn(val_predictions, y_val).item()  # 计算整集验证 loss；val_loss 是浮点数，没有 shape。

            train_errors = train_predictions - y_train_eval  # 计算训练集逐样本误差；train_errors 的 shape 是 (13209,)。
            val_errors = val_predictions - y_val  # 计算验证集逐样本误差；val_errors 的 shape 是 (3303,)。

            train_ss_res = torch.sum(train_errors ** 2)  # 计算训练集残差平方和；结果是 0 维标量张量，shape 是 ()。
            train_ss_tot = torch.sum((y_train_eval - torch.mean(y_train_eval)) ** 2)  # 计算训练集总平方和；结果是 0 维标量张量，shape 是 ()。
            train_r2 = (1 - train_ss_res / train_ss_tot).item()  # 计算训练集 R2；train_r2 是浮点数，没有 shape。

            val_ss_res = torch.sum(val_errors ** 2)  # 计算验证集残差平方和；结果是 0 维标量张量，shape 是 ()。
            val_ss_tot = torch.sum((y_val - torch.mean(y_val)) ** 2)  # 计算验证集总平方和；结果是 0 维标量张量，shape 是 ()。
            val_r2 = (1 - val_ss_res / val_ss_tot).item()  # 计算验证集 R2；val_r2 是浮点数，没有 shape。

        train_loss_history.append(train_loss)  # 记录当前 epoch 的训练 loss；列表长度每轮加 1。
        val_loss_history.append(val_loss)  # 记录当前 epoch 的验证 loss；列表长度每轮加 1。
        train_r2_history.append(train_r2)  # 记录当前 epoch 的训练 R2；列表长度每轮加 1。
        val_r2_history.append(val_r2)  # 记录当前 epoch 的验证 R2；列表长度每轮加 1。

        if epoch % 50 == 0 or epoch == epochs - 1:  # 每 50 轮和最后一轮打印一次日志；这是布尔判断，没有 shape。
            print(  # 打印当前训练日志；这里没有张量 shape。
                f"Epoch {epoch:03d} | "  # 打印 epoch 编号；这是字符串，没有 shape。
                f"Train Loss: {train_loss:.4f} | "  # 打印训练 loss；这是字符串，没有 shape。
                f"Val Loss: {val_loss:.4f} | "  # 打印验证 loss；这是字符串，没有 shape。
                f"Train R2: {train_r2:.4f} | "  # 打印训练 R2；这是字符串，没有 shape。
                f"Val R2: {val_r2:.4f}"  # 打印验证 R2；这是字符串，没有 shape。
            )

    net.eval()  # 把网络切到最终测试模式；这里没有张量 shape。
    with torch.no_grad():  # 关闭梯度，开始最终测试；这里没有张量 shape。
        test_predictions = net(x_test).squeeze(1)  # 对整个测试集做预测；squeeze 前 shape 是 (4128, 1)，squeeze 后 shape 是 (4128,)。
        test_errors = test_predictions - y_test  # 计算测试集逐样本误差；test_errors 的 shape 是 (4128,)。
        test_mse = torch.mean(test_errors ** 2).item()  # 计算测试集 MSE；test_mse 是浮点数，没有 shape。
        test_rmse = torch.sqrt(torch.mean(test_errors ** 2)).item()  # 计算测试集 RMSE；test_rmse 是浮点数，没有 shape。
        test_mae = torch.mean(torch.abs(test_errors)).item()  # 计算测试集 MAE；test_mae 是浮点数，没有 shape。
        test_ss_res = torch.sum(test_errors ** 2)  # 计算测试集残差平方和；结果是 0 维标量张量，shape 是 ()。
        test_ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)  # 计算测试集总平方和；结果是 0 维标量张量，shape 是 ()。
        test_r2 = (1 - test_ss_res / test_ss_tot).item()  # 计算测试集 R2；test_r2 是浮点数，没有 shape。

    print("\nFinal test metrics:")  # 打印测试集指标标题；这是字符串，没有 shape。
    print(f"MSE : {test_mse:.4f}")  # 打印测试集 MSE；这是字符串，没有 shape。
    print(f"RMSE: {test_rmse:.4f}")  # 打印测试集 RMSE；这是字符串，没有 shape。
    print(f"MAE : {test_mae:.4f}")  # 打印测试集 MAE；这是字符串，没有 shape。
    print(f"R2  : {test_r2:.4f}")  # 打印测试集 R2；这是字符串，没有 shape。

    print("\nFeature names:")  # 打印特征名标题；这是字符串，没有 shape。
    print(dataset.feature_names)  # 打印 8 个原始特征名组成的列表；列表长度是 8。

    print("\nSample inference results:")  # 打印样例推理标题；这是字符串，没有 shape。
    sample_predictions = test_predictions[:5]  # 取前 5 个测试预测值；sample_predictions 的 shape 是 (5,)。
    sample_targets = y_test[:5]  # 取前 5 个测试真实值；sample_targets 的 shape 是 (5,)。
    for idx, (pred_value, actual_value) in enumerate(zip(sample_predictions, sample_targets), start=1):  # 同时遍历 5 个预测值和真实值；pred_value 和 actual_value 都是 0 维标量张量，shape 是 ()。
        print(  # 打印每个样本的预测和真实值；这里没有张量 shape。
            f"Sample {idx}: "  # 打印样本编号；这是字符串，没有 shape。
            f"predicted={pred_value.item():.4f}, "  # 打印预测值；这是字符串，没有 shape。
            f"actual={actual_value.item():.4f}"  # 打印真实值；这是字符串，没有 shape。
        )

    epoch_axis = range(1, epochs + 1)  # 创建横轴 epoch 序列；这是 range 对象，没有张量 shape，长度是 300。
    plt.figure(figsize=(12, 5))  # 创建宽 12、高 5 的画布；这里没有张量 shape。

    plt.subplot(1, 2, 1)  # 选择第 1 个子图位置；这里没有张量 shape。
    plt.plot(epoch_axis, train_loss_history, label="Train Loss")  # 画训练 loss 曲线；横轴长度是 300，纵轴列表长度是 300。
    plt.plot(epoch_axis, val_loss_history, label="Val Loss")  # 画验证 loss 曲线；横轴长度是 300，纵轴列表长度是 300。
    plt.xlabel("Epoch")  # 设置横轴名字；这里没有张量 shape。
    plt.ylabel("MSE Loss")  # 设置纵轴名字；这里没有张量 shape。
    plt.title("Loss Curve")  # 设置标题；这里没有张量 shape。
    plt.legend()  # 显示图例；这里没有张量 shape。

    plt.subplot(1, 2, 2)  # 选择第 2 个子图位置；这里没有张量 shape。
    plt.plot(epoch_axis, train_r2_history, label="Train R2")  # 画训练 R2 曲线；横轴长度是 300，纵轴列表长度是 300。
    plt.plot(epoch_axis, val_r2_history, label="Val R2")  # 画验证 R2 曲线；横轴长度是 300，纵轴列表长度是 300。
    plt.xlabel("Epoch")  # 设置横轴名字；这里没有张量 shape。
    plt.ylabel("R2 Score")  # 设置纵轴名字；这里没有张量 shape。
    plt.title("R2 Curve")  # 设置标题；这里没有张量 shape。
    plt.legend()  # 显示图例；这里没有张量 shape。

    plt.tight_layout()  # 自动调整子图间距；这里没有张量 shape。
    plt.show()  # 显示图像窗口；这里没有张量 shape。
