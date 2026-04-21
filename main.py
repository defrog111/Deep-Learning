# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Simple linear regression training template with PyTorch."""  # 用一句话说明这个脚本的用途。

import os  # 导入 os，用来设置 Matplotlib 的缓存目录和后端相关环境变量。

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))  # 把 Matplotlib 缓存目录设到当前项目下，避免写入用户目录失败。

import matplotlib  # 导入 Matplotlib 主模块，用来切换无界面后端。
import torch  # 导入 PyTorch，用来定义模型、张量和训练流程。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，替代已废弃的 Boston 数据集。
from sklearn.model_selection import train_test_split  # 导入数据集切分工具，用于划分训练集和测试集。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，让特征更容易训练收敛。
from torch.utils.data import DataLoader, TensorDataset  # 导入批量加载工具，把数据拆成一个个 batch。

matplotlib.use("Agg")  # 使用无界面后端，避免在终端或沙箱环境里弹图时报图形连接错误。

import matplotlib.pyplot as plt  # 导入画图库，用来画 loss 曲线和 accuracy 曲线。


# 以后我们每次写模型代码都默认检查这些问题，并尽量直接落实到代码结构里。
# 1. 怎么加速: device 选择、batch size、混合精度、DataLoader 参数、模型复杂度。
# 2. 多模态吗: 输入是不是只有表格/图像/文本中的一种，后续是否要融合多种模态。
# 3. 用不用 PyTorch: 如果只是传统机器学习，sklearn 可能更直接；深度学习优先 PyTorch。
# 4. metric 要有哪些: 回归看 MSE/MAE/RMSE/R2，分类看 Accuracy/Precision/Recall/F1/AUC。
# 5. train/test 要严格分开: 预处理只在训练集 fit，评估时用 eval/no_grad。
# 6. loss 用什么: 回归常见 MSELoss/L1Loss/HuberLoss，分类常见 CrossEntropy/BCE。
# 7. 参数怎么调: 学习率、epoch、batch size、hidden size、optimizer、weight decay。
# 8. 代码里要留出问题清单: 让后续继续扩展时不漏掉加速、指标和任务定义。
#
# 当前这个例子是表格回归，不是多模态任务；如果你后面给我图像、文本、语音，我会主动把多模态方案一起写进代码。


class Net(torch.nn.Module):  # 定义一个继承自 PyTorch 模块的线性回归模型。
    def __init__(self, n_feature, n_output):  # 初始化模型时传入输入维度和输出维度。
        super().__init__()  # 调用父类初始化，让这个类具备 Module 的能力。
        self.linear = torch.nn.Linear(n_feature, n_output)  # 创建一个全连接层，输入 shape = (batch_size, n_feature)，输出 shape = (batch_size, 1)。
        # 如果后面发现模型欠拟合，可以把这里只放一层线性层改成多层 MLP。
        # 例如可以改成 Linear -> ReLU -> Linear，这样模型表达能力会更强。
        # 但模型更复杂以后，也要同步关注过拟合、训练时间和是否需要正则化。

    def forward(self, x):  # 定义前向传播逻辑，x 是输入特征。
        return self.linear(x)  # 直接把输入交给线性层，返回 shape = (batch_size, 1) 的预测结果。


def set_seed(seed: int = 42) -> None:  # 固定随机种子，让实验结果更稳定、更方便复现。
    torch.manual_seed(seed)  # 给 CPU 随机数生成器设种子，后面初始化参数时结果更稳定。
    if torch.cuda.is_available():  # 如果当前环境里有 CUDA GPU，就顺手把 GPU 的随机种子也固定住。
        torch.cuda.manual_seed_all(seed)  # 给所有 CUDA 设备设同一个种子，方便多卡或单卡复现实验。


def get_device():  # 自动选择可用设备，把加速入口集中管理。
    if torch.backends.mps.is_available():  # 先检查 Apple Silicon 的 MPS 后端是否可用。
        return torch.device("mps")  # 返回 MPS 设备对象，后面模型和 batch 都会放到这里。
    if torch.cuda.is_available():  # 如果没有 MPS，再检查 NVIDIA 的 CUDA 是否可用。
        return torch.device("cuda")  # 返回 CUDA 设备对象，后面训练会尽量走 GPU 加速。
    return torch.device("cpu")  # 如果都不可用，就回退到 CPU 上运行。
    # 加速时先检查 device 是否真的切到了 GPU/MPS，这通常是第一步。
    # 如果还是慢，再考虑增大 batch_size、减少日志、换更快的优化器或更简单的模型。


def regression_metrics(pred, target, threshold):  # 统一计算多个回归指标，避免只盯着单一 loss。
    abs_error = (pred - target).abs()  # 逐样本绝对误差，shape = (num_samples,)。
    mse = torch.mean((pred - target) ** 2)  # 均方误差，输出是 0 维标量张量，shape = ()。
    mae = torch.mean(abs_error)  # 平均绝对误差，输出是 0 维标量张量，shape = ()。
    rmse = torch.sqrt(mse)  # 对 MSE 开平方得到 RMSE，输出仍然是标量，shape = ()。
    accuracy = (abs_error < threshold).float().mean()  # 误差小于阈值记为 1，否则记为 0，最后取均值，shape = ()。
    total_var = torch.sum((target - target.mean()) ** 2)  # 目标值总方差分量，输出是标量，shape = ()。
    residual_var = torch.sum((target - pred) ** 2)  # 残差平方和，输出是标量，shape = ()。
    r2 = 1 - residual_var / total_var  # 根据 1 - SSR/SST 计算 R2，输出是标量，shape = ()。
    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "rmse": rmse.item(),
        "threshold_accuracy": accuracy.item(),
        "r2": r2.item(),
    }


def prepare_data(batch_size):  # 负责数据下载、切分、标准化和 DataLoader 构建。
    data = fetch_california_housing()  # 下载或读取加州房价数据集，里面包含特征矩阵和回归标签。
    x = data.data  # 取出输入特征矩阵，shape = (20640, 8)，每行 1 个样本、每列 1 个特征。
    y = data.target  # 取出每个样本对应的房价标签，shape = (20640,)，是一维回归目标。

    x_train, x_test, y_train, y_test = train_test_split(
        x,  # 原始全部特征，shape = (20640, 8)。
        y,  # 原始全部标签，shape = (20640,)。
        test_size=0.2,  # 抽出 20% 作为测试集，所以测试集大约有 4128 个样本。
        random_state=42,  # 固定切分随机种子，保证每次 train/test 划分一致。
    )

    scaler = StandardScaler()  # 创建标准化器对象，只用训练集来学习均值和方差。
    x_train = scaler.fit_transform(x_train)  # 在训练集上先拟合再标准化，shape 仍然是 (16512, 8) 左右。
    x_test = scaler.transform(x_test)  # 用训练集得到的标准化规则去转换测试集，shape 仍然是 (4128, 8) 左右。

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),  # 把训练特征转成 float32 张量，shape = (16512, 8) 左右。
        torch.tensor(y_train, dtype=torch.float32),  # 把训练标签转成 float32 张量，shape = (16512,) 左右。
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),  # 把测试特征转成 float32 张量，shape = (4128, 8) 左右。
        torch.tensor(y_test, dtype=torch.float32),  # 把测试标签转成 float32 张量，shape = (4128,) 左右。
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练加载器里每个 batch 的 batch_x shape 约是 (batch_size, 8)，batch_y shape 约是 (batch_size,)。
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试加载器里每个 batch 的 shape 也类似，只是不打乱顺序。
    # batch_size 调大: 通常吞吐更高、训练更快，但更吃显存，太大还可能影响泛化。
    # batch_size 调小: 更省显存，梯度噪声更大，有时泛化更好，但训练可能更慢。
    # 如果后面数据量更大，可以继续考虑 DataLoader 的 num_workers、pin_memory 等加速参数。
    input_dim = x_train.shape[1]  # 取输入特征维度，当前是 8，后面创建线性层时会用到。
    return train_loader, test_loader, input_dim  # 返回训练加载器、测试加载器和输入维度。


def train_one_epoch(model, train_loader, loss_func, optimizer, device):  # 单独封装一轮训练逻辑，让主函数更清楚。
    model.train()  # 明确切换到训练模式，后面如果加 Dropout 或 BatchNorm 才不会出错。
    total_loss = 0.0  # 记录这一整轮里所有 batch 的损失总和，Python 浮点数，没有 shape 概念。

    for batch_x, batch_y in train_loader:  # 从 DataLoader 取出一个 batch，batch_x shape 约是 (batch_size, 8)，batch_y shape 约是 (batch_size,)。
        batch_x = batch_x.to(device)  # 把当前 batch 的特征移动到目标设备上，shape 不变，仍约是 (batch_size, 8)。
        batch_y = batch_y.to(device)  # 把当前 batch 的标签移动到目标设备上，shape 不变，仍约是 (batch_size,)。
        pred = model(batch_x).squeeze(1)  # 先经过线性层得到 (batch_size, 1)，再 squeeze 后变成 (batch_size,)。
        loss = loss_func(pred, batch_y)  # 计算当前 batch 的损失，输出是标量张量，shape = ()。

        optimizer.zero_grad()  # 在反向传播前先清空旧梯度，否则不同 batch 的梯度会累加。
        loss.backward()  # 根据当前 batch 的损失做反向传播，把梯度写进每个参数的 .grad。
        optimizer.step()  # 使用优化器按梯度方向更新参数，让模型朝更小 loss 的方向移动。
        # 如果训练不稳定、loss 上下乱跳，优先先试减小 learning_rate。
        # 如果收敛太慢，可以尝试增大学习率一点点，或者从 SGD 改成 Adam / AdamW。

        total_loss += loss.item()  # 把当前 batch 的标量损失转成 Python 数字并累加起来。

    return total_loss / len(train_loader)  # 返回这一轮训练集平均 loss。


def evaluate(model, data_loader, loss_func, device, accuracy_threshold):  # 单独封装评估逻辑，让 train/test 严格分开。
    model.eval()  # 明确切换到测试模式，保证评估和训练严格分开。
    total_loss = 0.0  # 记录所有测试 batch 的损失总和，最后会再除以 batch 数。
    pred_batches = []  # 用列表保存每个 batch 的预测结果，每个元素 shape 约是 (batch_size,)。
    target_batches = []  # 用列表保存每个 batch 的真实标签，每个元素 shape 约是 (batch_size,)。

    with torch.no_grad():  # 进入不计算梯度模式，评估时更省内存也更快。
        for batch_x, batch_y in data_loader:  # 从测试加载器逐批取数据，shape 规则和训练时一致。
            batch_x = batch_x.to(device)  # 把当前测试 batch 的特征移动到目标设备上，shape 约是 (batch_size, 8)。
            batch_y = batch_y.to(device)  # 把当前测试 batch 的标签移动到目标设备上，shape 约是 (batch_size,)。
            pred = model(batch_x).squeeze(1)  # 前向传播后得到当前 batch 的预测值，shape 约是 (batch_size,)。
            loss = loss_func(pred, batch_y)  # 计算当前测试 batch 的损失，输出是标量张量，shape = ()。

            total_loss += loss.item()  # 把当前 batch 的损失加入总和。
            pred_batches.append(pred)  # 追加当前 batch 的预测张量，单个元素 shape 约是 (batch_size,)。
            target_batches.append(batch_y)  # 追加当前 batch 的真实标签张量，单个元素 shape 约是 (batch_size,)。

    all_pred = torch.cat(pred_batches)  # 把所有 batch 的预测在第 0 维拼起来，最终 shape = (num_test_samples,)。
    all_target = torch.cat(target_batches)  # 把所有 batch 的标签在第 0 维拼起来，最终 shape = (num_test_samples,)。
    metrics = regression_metrics(all_pred, all_target, accuracy_threshold)  # 基于整份测试集计算回归指标。
    metrics["loss"] = total_loss / len(data_loader)  # 额外记录平均测试损失，这是一个 Python 浮点数。
    return metrics  # 返回一个字典，里面有 loss、mae、rmse、r2、threshold_accuracy 等指标。


def plot_curves(train_loss_history, test_loss_history, test_accuracy_history, plot_path):  # 把画图逻辑单独拆出来，主流程更干净。
    epochs = range(1, len(train_loss_history) + 1)  # 生成横轴 epoch 序列，长度等于训练轮数。
    plt.figure(figsize=(12, 5))  # 创建一个宽 12、高 5 的画布，让两张图并排显示。

    plt.subplot(1, 2, 1)  # 选择左边第 1 张子图，专门画 loss 曲线。
    plt.plot(epochs, train_loss_history, label="Train Loss")  # 训练 loss 列表长度是 num_epochs，没有张量 shape 概念。
    plt.plot(epochs, test_loss_history, label="Test Loss")  # 测试 loss 列表长度同样是 num_epochs。
    plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
    plt.ylabel("Loss")  # 设置纵轴名称为 Loss。
    plt.title("Loss Curve")  # 给左边图设置标题。
    plt.legend()  # 显示图例，区分训练集和测试集曲线。

    plt.subplot(1, 2, 2)  # 选择右边第 2 张子图，专门画自定义 accuracy 曲线。
    plt.plot(epochs, test_accuracy_history, label="Test Accuracy", color="orange")  # 测试 accuracy 列表长度是 num_epochs。
    plt.xlabel("Epoch")  # 设置横轴名称为 Epoch。
    plt.ylabel("Accuracy")  # 设置纵轴名称为 Accuracy。
    plt.title("Accuracy Curve")  # 给右边图设置标题。
    plt.legend()  # 显示图例，说明这条线表示测试 accuracy。

    plt.tight_layout()  # 自动调整子图间距，避免标题和坐标轴文字重叠。
    plt.savefig(plot_path, dpi=200)  # 把图直接保存成图片文件，避免 plt.show() 依赖图形界面。


def main():  # 主入口函数，后面你继续扩展模型时就改这里和对应子函数。
    config = {
        "seed": 42,  # 随机种子，控制数据切分和参数初始化的稳定性。
        "batch_size": 32,  # 每个 batch 放 32 个样本，所以 batch_x shape 通常是 (32, 8)；想加速时常先试着调大它。
        "learning_rate": 0.01,  # 学习率，控制每次参数更新的步长大小；太大可能震荡，太小可能学得很慢。
        "num_epochs": 10,  # 总训练轮数，表示完整遍历训练集 10 次；如果 train/test 都还没收敛，可以继续增大。
        "accuracy_threshold": 0.5,  # 自定义“回归 accuracy”阈值，误差小于 0.5 视为预测正确；阈值大小会直接影响这个指标高低。
        "loss_name": "MSELoss",  # 当前回归任务使用 MSELoss；如果异常值多，可试 HuberLoss；更关心绝对误差时可试 L1Loss。
        "plot_path": "training_curves.png",  # 训练曲线图片的保存路径。
    }
    # 这份 config 是后面调参最先看的地方。
    # 常见调参顺序:
    # 1. 先看 learning_rate，因为它最容易让训练“完全不学”或“乱跳”。
    # 2. 再看 batch_size，因为它影响速度、显存和梯度稳定性。
    # 3. 再看 num_epochs，因为轮数太少时你可能只是还没训练够。
    # 4. 然后考虑换 optimizer、loss，最后再增加模型复杂度。

    set_seed(config["seed"])  # 每次运行前先固定随机种子，减少实验波动。
    device = get_device()  # 统一管理 device 选择，是后面做加速时最先检查的点。
    print(f"Using device: {device}")  # 打印当前实际使用的设备，方便确认是否成功切到 GPU/MPS。

    train_loader, test_loader, input_dim = prepare_data(config["batch_size"])  # 训练加载器和测试加载器的单个 batch shape 都大致是 ((32, 8), (32,))。
    model = Net(input_dim, 1).to(device)  # 创建模型后放到目标设备上，线性层权重 shape = (1, 8)，偏置 shape = (1,)。
    loss_func = getattr(torch.nn, config["loss_name"])()  # 实例化损失函数对象，输入 pred 和 target 都要求 shape 对齐。
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])  # 创建优化器，负责根据梯度更新线性层参数。
    # SGD 往往更基础、更适合教学理解；如果你想更快收敛，经常会尝试 Adam 或 AdamW。
    # Adam / AdamW 一般更容易训起来，但也不是任何任务都一定更好，所以还是要看 test 指标。

    train_loss_history = []  # 记录每一轮训练集平均 loss，后面用来画图。
    test_loss_history = []  # 记录每一轮测试集 loss，后面用来画图。
    test_accuracy_history = []  # 记录每一轮测试集上的自定义 accuracy，后面用来画图。

    for epoch in range(config["num_epochs"]):  # 外层循环控制训练轮数，epoch 依次是 0 到 num_epochs - 1。
        train_loss = train_one_epoch(model, train_loader, loss_func, optimizer, device)  # 训练完一轮后返回平均训练损失，是 Python 浮点数。
        test_metrics = evaluate(model, test_loader, loss_func, device, config["accuracy_threshold"])  # 在完整测试集上评估，返回指标字典。

        train_loss_history.append(train_loss)  # 把当前轮训练损失追加到列表中，列表长度逐轮增加。
        test_loss_history.append(test_metrics["loss"])  # 把当前轮测试损失追加到列表中，后面画 loss 曲线会用到。
        test_accuracy_history.append(test_metrics["threshold_accuracy"])  # 把当前轮测试 accuracy 追加到列表中，后面画 accuracy 曲线会用到。

        print(
            f"Epoch {epoch}, "  # 打印当前是第几轮训练。
            f"Train Loss: {train_loss:.4f}, "  # 打印训练集平均损失。
            f"Test Loss: {test_metrics['loss']:.4f}, "  # 打印测试集平均损失。
            f"MAE: {test_metrics['mae']:.4f}, "  # 打印平均绝对误差。
            f"RMSE: {test_metrics['rmse']:.4f}, "  # 打印均方根误差。
            f"R2: {test_metrics['r2']:.4f}, "  # 打印拟合优度。
            f"Test Accuracy: {test_metrics['threshold_accuracy']:.4f}"  # 打印阈值型回归 accuracy。
        )

    final_metrics = evaluate(model, test_loader, loss_func, device, config["accuracy_threshold"])  # 最终统一输出一组完整回归指标。
    print("\nTest Loss:", final_metrics["loss"])  # 打印最终测试损失，帮助判断整体拟合情况。
    print("Test MAE:", final_metrics["mae"])  # 打印最终 MAE，帮助判断平均偏差大小。
    print("Test RMSE:", final_metrics["rmse"])  # 打印最终 RMSE，帮助判断大误差是否明显。
    print("Test R2:", final_metrics["r2"])  # 打印最终 R2，帮助判断模型解释目标方差的能力。
    print("Test Accuracy:", final_metrics["threshold_accuracy"])  # 打印最终阈值型回归 accuracy。
    # 看结果时不要只盯着一个指标。
    # 回归任务通常至少同时看 loss、MAE、RMSE、R2。
    # 如果 train 指标很好但 test 指标差，通常要怀疑过拟合、数据泄漏或训练/测试分布差异。

    plot_curves(train_loss_history, test_loss_history, test_accuracy_history, config["plot_path"])  # 根据历史列表画出训练过程曲线。
    print(f"Saved plot to: {config['plot_path']}")  # 打印图片保存路径，方便确认脚本完整执行结束。


if __name__ == "__main__":  # 直接运行这个文件时才执行 main，被别的文件导入时不会自动训练。
    main()
