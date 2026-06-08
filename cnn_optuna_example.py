# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便写中文注释。
"""CNN + Optuna hyperparameter tuning example on sklearn digits."""  # 这个脚本演示用 Optuna 调 CNN 超参数。

import optuna  # 导入 Optuna，用来做自动超参数搜索；如果没有安装，运行 pip install optuna。
import torch  # 导入 PyTorch，用来定义 CNN、张量、损失函数和优化器。
from sklearn.datasets import load_digits  # 导入 sklearn 自带 8x8 手写数字数据集，不需要联网下载。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分 train/val/test。
from torch.utils.data import DataLoader, TensorDataset  # 导入 Dataset 和 DataLoader，方便按 batch 训练。


torch.manual_seed(42)  # 固定随机种子，让每次运行更稳定。


class SmallCNN(torch.nn.Module):  # 定义一个小 CNN 分类器，输入是 1 通道 8x8 图片。
    def __init__(self, channels: int, dropout: float) -> None:  # channels 和 dropout 由 Optuna 调参决定。
        super().__init__()  # 调用父类初始化。
        self.features = torch.nn.Sequential(  # 定义卷积特征提取部分。
            torch.nn.Conv2d(1, channels, kernel_size=3, padding=1),  # 输入 shape = (B, 1, 8, 8)，输出 shape = (B, C, 8, 8)。
            torch.nn.ReLU(),  # 非线性激活，shape 不变。
            torch.nn.MaxPool2d(2),  # 下采样，输出 shape = (B, C, 4, 4)。
            torch.nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),  # 输出 shape = (B, 2C, 4, 4)。
            torch.nn.ReLU(),  # 非线性激活，shape 不变。
            torch.nn.MaxPool2d(2),  # 再下采样，输出 shape = (B, 2C, 2, 2)。
        )  # 卷积模块结束。
        self.classifier = torch.nn.Sequential(  # 定义分类头。
            torch.nn.Flatten(),  # 展平，shape = (B, 2C*2*2)。
            torch.nn.Dropout(dropout),  # 随机丢弃一部分特征，降低过拟合。
            torch.nn.Linear(channels * 2 * 2 * 2, 10),  # 输出 10 类 logits，shape = (B, 10)。
        )  # 分类头结束。

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播，输入 x shape = (B, 1, 8, 8)。
        x = self.features(x)  # 经过卷积层提取特征，shape = (B, 2C, 2, 2)。
        logits = self.classifier(x)  # 经过分类头得到 logits，shape = (B, 10)。
        return logits  # 返回未经过 softmax 的分类分数。


def accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:  # 计算分类准确率。
    model.eval()  # 切换到评估模式，关闭 Dropout 的随机行为。
    correct = 0  # 统计预测正确样本数。
    total = 0  # 统计总样本数。

    with torch.no_grad():  # 评估阶段不需要梯度。
        for batch_x, batch_y in loader:  # 遍历验证或测试集 batch。
            batch_x = batch_x.to(device)  # 图片搬到 device，shape = (B, 1, 8, 8)。
            batch_y = batch_y.to(device)  # 标签搬到 device，shape = (B,)。
            logits = model(batch_x)  # 前向传播，shape = (B, 10)。
            preds = torch.argmax(logits, dim=1)  # 取最大 logit 的类别，shape = (B,)。
            correct += (preds == batch_y).sum().item()  # 累加预测正确数量。
            total += batch_y.shape[0]  # 累加当前 batch 样本数。

    return correct / total  # 返回准确率。


def train_one_epoch(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:  # 训练一轮并返回平均 loss。
    model.train()  # 切换到训练模式，Dropout 会启用。
    total_loss = 0.0  # 累加所有 batch 的 loss。
    total = 0  # 累加训练样本数。

    for batch_x, batch_y in loader:  # 遍历训练集 batch。
        batch_x = batch_x.to(device)  # 图片搬到 device，shape = (B, 1, 8, 8)。
        batch_y = batch_y.to(device)  # 标签搬到 device，shape = (B,)。
        logits = model(batch_x)  # 前向传播，shape = (B, 10)。
        loss = loss_fn(logits, batch_y)  # 计算交叉熵损失，shape = ()。
        optimizer.zero_grad()  # 清空上一轮梯度。
        loss.backward()  # 反向传播。
        optimizer.step()  # 更新参数。
        total_loss += loss.item() * batch_x.shape[0]  # 按样本数累加 loss。
        total += batch_x.shape[0]  # 累加当前 batch 样本数。

    return total_loss / total  # 返回训练集平均 loss。


def validate(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> tuple[float, float]:  # 在验证集或测试集上评估 loss 和 accuracy。
    model.eval()  # 切换到评估模式，Dropout 会关闭。
    total_loss = 0.0  # 累加验证 loss。
    correct = 0  # 统计预测正确样本数。
    total = 0  # 统计总样本数。

    with torch.no_grad():  # 验证阶段不需要梯度。
        for batch_x, batch_y in loader:  # 遍历验证集或测试集 batch。
            batch_x = batch_x.to(device)  # 图片搬到 device，shape = (B, 1, 8, 8)。
            batch_y = batch_y.to(device)  # 标签搬到 device，shape = (B,)。
            logits = model(batch_x)  # 前向传播，shape = (B, 10)。
            loss = loss_fn(logits, batch_y)  # 计算当前 batch loss，shape = ()。
            preds = torch.argmax(logits, dim=1)  # 取最大 logit 的类别，shape = (B,)。
            total_loss += loss.item() * batch_x.shape[0]  # 按样本数累加 loss。
            correct += (preds == batch_y).sum().item()  # 累加预测正确数量。
            total += batch_y.shape[0]  # 累加当前 batch 样本数。

    return total_loss / total, correct / total  # 返回平均 loss 和 accuracy。


def run_inference(model: torch.nn.Module, image: torch.Tensor, device: torch.device) -> tuple[int, float, torch.Tensor]:  # 对单张图片做 inference。
    model.eval()  # 切换到评估模式。
    with torch.no_grad():  # 推理阶段不需要梯度。
        image = image.unsqueeze(0).to(device)  # 给单张图片加 batch 维度，shape = (1, 1, 8, 8)。
        logits = model(image)  # 前向传播得到 logits，shape = (1, 10)。
        probs = torch.softmax(logits, dim=1).squeeze(0)  # 转成类别概率并去掉 batch 维，shape = (10,)。
        pred = torch.argmax(probs).item()  # 取概率最大的类别，Python int。
        confidence = probs[pred].item()  # 取预测类别的概率，Python float。
    return pred, confidence, probs.cpu()  # 返回预测类别、置信度和完整概率向量。


def build_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, torch.Tensor]:  # 根据 batch_size 构造 DataLoader，并返回一个推理样本。
    digits = load_digits()  # 读取数据集，图片原始 shape = (1797, 8, 8)。
    x = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0  # 转成张量并归一化，shape = (1797, 1, 8, 8)。
    y = torch.tensor(digits.target, dtype=torch.long)  # 转成分类标签，shape = (1797,)。

    x_train_full, x_test, y_train_full, y_test = train_test_split(  # 先切出测试集。
        x,  # 全部图片张量，shape = (1797, 1, 8, 8)。
        y,  # 全部标签张量，shape = (1797,)。
        test_size=0.2,  # 20% 作为测试集。
        random_state=42,  # 固定切分随机种子。
        stratify=y,  # 按类别比例分层抽样。
    )  # 切分后训练+验证约 1437 张，测试约 360 张。
    x_train, x_val, y_train, y_val = train_test_split(  # 再切出验证集。
        x_train_full,  # 训练+验证图片。
        y_train_full,  # 训练+验证标签。
        test_size=0.2,  # 20% 的训练+验证作为验证集。
        random_state=42,  # 固定切分随机种子。
        stratify=y_train_full,  # 按类别比例分层抽样。
    )  # 切分后训练约 1149 张，验证约 288 张。

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)  # 训练集 loader。
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=256, shuffle=False)  # 验证集 loader。
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=256, shuffle=False)  # 测试集 loader。
    inference_image = x_test[0]  # 取一张测试图片做 inference demo，shape = (1, 8, 8)。
    inference_label = y_test[0]  # 取对应真实标签，shape = ()。
    return train_loader, val_loader, test_loader, inference_image, inference_label  # 返回 DataLoader 和单样本推理数据。


def objective(trial: optuna.Trial) -> float:  # Optuna 每个 trial 都会调用一次这个目标函数。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 CPU 或 GPU。
    channels = trial.suggest_categorical("channels", [8, 16, 32])  # 搜索卷积通道数。
    dropout = trial.suggest_float("dropout", 0.0, 0.5)  # 搜索 Dropout 比例。
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)  # 搜索学习率，log=True 表示按数量级搜索。
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)  # 搜索 L2 正则强度。
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # 搜索 batch size。
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])  # 搜索优化器类型。

    train_loader, val_loader, _, _, _ = build_loaders(batch_size)  # 按当前 batch_size 构造数据加载器。
    model = SmallCNN(channels=channels, dropout=dropout).to(device)  # 创建当前 trial 的 CNN 模型。
    loss_fn = torch.nn.CrossEntropyLoss()  # 多分类交叉熵，输入 logits shape = (B, 10)，标签 shape = (B,)。

    if optimizer_name == "SGD":  # 如果 Optuna 选中了 SGD。
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)  # 定义 SGD 优化器。
    elif optimizer_name == "AdamW":  # 如果 Optuna 选中了 AdamW。
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)  # 定义 AdamW 优化器。
    else:  # 否则使用 Adam。
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 定义 Adam 优化器。

    for epoch in range(12):  # 每个 trial 训练 12 轮，教学例子不要太慢。
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)  # 训练一轮，返回训练 loss。
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)  # 每轮结束后计算验证集 loss 和准确率。
        trial.set_user_attr(f"epoch_{epoch}_train_loss", train_loss)  # 把训练 loss 记录到 trial 属性里，方便回看。
        trial.set_user_attr(f"epoch_{epoch}_val_loss", val_loss)  # 把验证 loss 记录到 trial 属性里，方便回看。
        trial.report(val_acc, epoch)  # 把中间结果报告给 Optuna，方便剪枝。

        if trial.should_prune():  # 如果当前 trial 明显不如其他 trial，就提前停止。
            raise optuna.TrialPruned()  # 抛出剪枝异常，Optuna 会记录为 pruned。

    return val_acc  # Optuna 会最大化这个验证准确率。


def main() -> None:  # 主函数，负责创建 study、启动搜索、最终测试。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 CPU 或 GPU。
    study = optuna.create_study(direction="maximize")  # 创建一个最大化目标的 Optuna study。
    study.optimize(objective, n_trials=20)  # 运行 20 次 trial，真实项目可以改成 50、100 或更多。

    print("\nBest trial:")  # 打印最优 trial 标题。
    print("value:", study.best_value)  # 打印最优验证集准确率。
    print("params:", study.best_params)  # 打印最优超参数组合。

    best = study.best_params  # 取出最优参数字典。
    train_loader, val_loader, test_loader, inference_image, inference_label = build_loaders(best["batch_size"])  # 用最优 batch_size 重建数据加载器。
    model = SmallCNN(channels=best["channels"], dropout=best["dropout"]).to(device)  # 用最优结构创建最终模型。
    loss_fn = torch.nn.CrossEntropyLoss()  # 定义最终训练的损失函数。

    if best["optimizer"] == "SGD":  # 根据最优 optimizer 名称创建优化器。
        optimizer = torch.optim.SGD(model.parameters(), lr=best["lr"], momentum=0.9, weight_decay=best["weight_decay"])
    elif best["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"], weight_decay=best["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"], weight_decay=best["weight_decay"])

    for epoch in range(20):  # 用最优参数重新训练最终模型 20 轮。
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)  # 训练一轮并返回训练 loss。
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)  # 在验证集上监控最终模型表现。
        if epoch % 5 == 0 or epoch == 19:  # 每 5 轮和最后一轮打印一次训练/验证日志。
            print(f"final train epoch {epoch:02d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val acc: {val_acc:.4f}")  # 打印最终训练日志。

    test_loss, test_acc = validate(model, test_loader, loss_fn, device)  # 在测试集上评估最终模型。
    pred, confidence, probs = run_inference(model, inference_image, device)  # 对单张测试图片做 inference。
    print("final test loss:", test_loss)  # 打印最终测试 loss。
    print("final test accuracy:", test_acc)  # 打印最终测试准确率。
    print("inference true label:", int(inference_label.item()))  # 打印单样本真实标签。
    print("inference pred label:", pred)  # 打印单样本预测标签。
    print("inference confidence:", confidence)  # 打印预测类别置信度。
    print("inference probs:", [round(value, 4) for value in probs.tolist()])  # 打印 10 个类别的概率。


if __name__ == "__main__":  # 只有直接运行这个文件时才执行 main。
    main()  # 启动 CNN + Optuna 调参流程。
