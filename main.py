# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Simple linear regression training template without PyTorch."""  # 用一句话说明这个脚本的用途。

import os  # 导入 os，用来设置 Matplotlib 的缓存目录和后端相关环境变量。

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))  # 把 Matplotlib 缓存目录设到当前项目下，避免写入用户目录失败。

import matplotlib  # 导入 Matplotlib 主模块，用来切换无界面后端。
import matplotlib.pyplot as plt  # 导入画图库，用来画 loss 曲线和 accuracy 曲线。
import numpy as np  # 导入 NumPy，用来做矩阵运算、手写线性回归和梯度下降。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，替代已废弃的 Boston 数据集。
from sklearn.model_selection import train_test_split  # 导入数据集切分工具，用于划分训练集和测试集。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，让特征更容易训练收敛。

matplotlib.use("Agg")  # 使用无界面后端，避免在终端或沙箱环境里弹图时报图形连接错误。


# 以后我们每次写模型代码都默认检查这些问题，并尽量直接落实到代码结构里。
# 1. 怎么加速: 向量化计算、batch size、减少不必要的 Python 循环、必要时换更快库。
# 2. 多模态吗: 输入是不是只有表格/图像/文本中的一种，后续是否要融合多种模态。
# 3. 用不用 PyTorch: 如果只是传统线性模型或基础机器学习，NumPy/sklearn 可能更直接。
# 4. metric 要有哪些: 回归看 MSE/MAE/RMSE/R2，分类看 Accuracy/Precision/Recall/F1/AUC。
# 5. train/test 要严格分开: 预处理只在训练集 fit，评估时不能混用训练信息。
# 6. loss 用什么: 回归常见 MSE、MAE、Huber；分类常见交叉熵。
# 7. 参数怎么调: 学习率、epoch、batch size、是否加偏置、模型复杂度。
# 8. 代码里要留出问题清单: 让后续继续扩展时不漏掉加速、指标和任务定义。
#
# 当前这个例子是表格回归，而且明确是不使用 PyTorch 的 NumPy 手写版本。


def set_seed(seed: int = 42) -> None:  # 固定随机种子，让打乱数据的顺序和实验结果更稳定。
    np.random.seed(seed)  # 给 NumPy 的随机数生成器设种子，后面 permutation 的结果会更可复现。


def regression_metrics(pred, target, threshold):  # 统一计算多个回归指标，避免只盯着单一 loss。
    abs_error = np.abs(pred - target)  # 逐样本绝对误差，shape = (num_samples,)。
    mse = np.mean((pred - target) ** 2)  # 均方误差，输出是 NumPy 标量，没有数组 shape。
    mae = np.mean(abs_error)  # 平均绝对误差，输出是 NumPy 标量。
    rmse = np.sqrt(mse)  # 对 MSE 开平方得到 RMSE，输出也是 NumPy 标量。
    accuracy = np.mean(abs_error < threshold)  # 误差小于阈值记为 True，最后取均值得到比例。
    total_var = np.sum((target - np.mean(target)) ** 2)  # 目标值总方差分量，输出是 NumPy 标量。
    residual_var = np.sum((target - pred) ** 2)  # 残差平方和，输出是 NumPy 标量。
    r2 = 1 - residual_var / total_var  # 根据 1 - SSR/SST 计算 R2，输出是 NumPy 标量。
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "threshold_accuracy": float(accuracy),
        "r2": float(r2),
    }


def prepare_data():  # 负责数据下载、切分、标准化和类型转换。
    data = fetch_california_housing()  # 下载或读取加州房价数据集，里面包含特征矩阵和回归标签。
    x = data.data  # 取出输入特征矩阵，shape = (20640, 8)，每行 1 个样本、每列 1 个特征。
    y = data.target  # 取出每个样本对应的房价标签，shape = (20640,)。

    x_train, x_test, y_train, y_test = train_test_split(
        x,  # 原始全部特征，shape = (20640, 8)。
        y,  # 原始全部标签，shape = (20640,)。
        test_size=0.2,  # 抽出 20% 作为测试集，所以测试集大约有 4128 个样本。
        random_state=42,  # 固定切分随机种子，保证每次 train/test 划分一致。
    )

    scaler = StandardScaler()  # 创建标准化器对象，只用训练集来学习均值和方差。
    x_train = scaler.fit_transform(x_train)  # 在训练集上先拟合再标准化，shape 仍然是 (16512, 8) 左右。
    x_test = scaler.transform(x_test)  # 用训练集得到的标准化规则去转换测试集，shape 仍然是 (4128, 8) 左右。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32 类型数组，shape = (16512, 8) 左右。
    y_train = y_train.astype(np.float32)  # 把训练标签转成 float32 类型数组，shape = (16512,) 左右。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32 类型数组，shape = (4128, 8) 左右。
    y_test = y_test.astype(np.float32)  # 把测试标签转成 float32 类型数组，shape = (4128,) 左右。

    input_dim = x_train.shape[1]  # 读取输入特征维度，当前是 8，后面初始化权重时会用到。
    return x_train, y_train, x_test, y_test, input_dim  # 返回训练集、测试集和输入维度。


def initialize_parameters(input_dim):  # 初始化线性回归参数，后面训练时会不断更新它们。
    w = np.zeros(input_dim, dtype=np.float32)  # 初始化权重向量，shape = (input_dim,)；当前大约是 (8,)。
    b = np.float32(0.0)  # 初始化偏置项，它是一个标量，不是向量。
    return w, b  # 返回初始权重和偏置。


def train_one_epoch(x_train, y_train, w, b, batch_size, learning_rate):  # 单独封装一轮训练逻辑，让主函数更清楚。
    indices = np.random.permutation(len(x_train))  # 生成训练样本的随机顺序，作用和 DataLoader 的 shuffle=True 类似。
    x_shuffled = x_train[indices]  # 按随机顺序重排训练特征，shape 仍然是 (num_train_samples, 8)。
    y_shuffled = y_train[indices]  # 按相同顺序重排训练标签，shape 仍然是 (num_train_samples,)。
    total_loss = 0.0  # 记录这一整轮里所有 batch 的损失总和，Python 浮点数，没有 shape 概念。
    num_batches = 0  # 记录这一轮总共处理了多少个 batch，后面用来算平均 loss。

    for start in range(0, len(x_shuffled), batch_size):  # 每次从打乱后的数据里切出一个 batch。
        end = start + batch_size  # 计算当前 batch 的结束位置。
        batch_x = x_shuffled[start:end]  # 取出当前 batch 的特征，shape 约是 (batch_size, 8)。
        batch_y = y_shuffled[start:end]  # 取出当前 batch 的标签，shape 约是 (batch_size,)。

        pred = batch_x @ w + b  # 用线性回归公式做前向传播，输出 shape 约是 (batch_size,)。
        error = pred - batch_y  # 计算预测值和真实值的差，shape 约是 (batch_size,)。
        loss = np.mean(error ** 2)  # 计算当前 batch 的均方误差损失，输出是 NumPy 标量。

        grad_w = (2.0 / len(batch_x)) * (batch_x.T @ error)  # 手动计算损失对权重的梯度，shape = (input_dim,)。
        grad_b = (2.0 / len(batch_x)) * np.sum(error)  # 手动计算损失对偏置的梯度，它是 NumPy 标量。

        w -= learning_rate * grad_w  # 按梯度下降公式更新权重；如果 learning_rate 太大会震荡，太小会收敛很慢。
        b -= learning_rate * grad_b  # 按梯度下降公式更新偏置；更新逻辑和权重一样，只是它是标量。

        total_loss += float(loss)  # 把当前 batch 的损失转成 Python 数字并累加起来。
        num_batches += 1  # 当前 batch 处理完成，把 batch 数量加 1。

    avg_loss = total_loss / num_batches  # 计算这一轮所有 batch 的平均损失。
    return w, b, avg_loss  # 返回更新后的参数和这一轮训练集平均 loss。


def evaluate(x_data, y_data, w, b, accuracy_threshold):  # 单独封装评估逻辑，让 train/test 严格分开。
    pred = x_data @ w + b  # 用当前线性模型做预测，x_data shape = (num_samples, 8)，pred shape = (num_samples,)。
    error = pred - y_data  # 计算预测误差，shape = (num_samples,)。
    metrics = regression_metrics(pred, y_data, accuracy_threshold)  # 基于整份数据统一计算回归指标。
    metrics["loss"] = float(np.mean(error ** 2))  # 额外记录平均 MSE 损失，这是 Python 浮点数。
    return metrics  # 返回一个字典，里面有 loss、mae、rmse、r2、threshold_accuracy 等指标。


def plot_curves(train_loss_history, test_loss_history, test_accuracy_history, plot_path):  # 把画图逻辑单独拆出来，主流程更干净。
    epochs = range(1, len(train_loss_history) + 1)  # 生成横轴 epoch 序列，长度等于训练轮数。
    plt.figure(figsize=(12, 5))  # 创建一个宽 12、高 5 的画布，让两张图并排显示。

    plt.subplot(1, 2, 1)  # 选择左边第 1 张子图，专门画 loss 曲线。
    plt.plot(epochs, train_loss_history, label="Train Loss")  # 训练 loss 列表长度是 num_epochs，没有数组 shape 概念。
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
        "seed": 42,  # 随机种子，控制数据切分和每一轮 shuffle 的稳定性。
        "batch_size": 32,  # 每个 batch 放 32 个样本，所以 batch_x shape 通常是 (32, 8)；想加速时可以先试着调大它。
        "learning_rate": 0.01,  # 学习率，控制每次参数更新的步长大小；太大可能震荡，太小可能学得很慢。
        "num_epochs": 100,  # 总训练轮数，表示完整遍历训练集 100 次；如果 train/test 都还没收敛，可以继续增大。
        "accuracy_threshold": 0.5,  # 自定义“回归 accuracy”阈值，误差小于 0.5 视为预测正确；阈值大小会直接影响这个指标高低。
        "plot_path": "training_curves.png",  # 训练曲线图片的保存路径。
    }
    # 这份 config 是后面调参最先看的地方。
    # 常见调参顺序:
    # 1. 先看 learning_rate，因为它最容易让训练“完全不学”或“乱跳”。
    # 2. 再看 batch_size，因为它影响速度、内存和梯度稳定性。
    # 3. 再看 num_epochs，因为轮数太少时你可能只是还没训练够。
    # 4. 然后考虑更复杂的模型，或者直接换到 sklearn / PyTorch 版本。

    set_seed(config["seed"])  # 每次运行前先固定随机种子，减少实验波动。

    x_train, y_train, x_test, y_test, input_dim = prepare_data()  # 准备训练集和测试集，并保证预处理只在训练集 fit。
    w, b = initialize_parameters(input_dim)  # 初始化线性回归参数，w shape = (8,)，b 是标量。

    train_loss_history = []  # 记录每一轮训练集平均 loss，后面用来画图。
    test_loss_history = []  # 记录每一轮测试集 loss，后面用来画图。
    test_accuracy_history = []  # 记录每一轮测试集上的自定义 accuracy，后面用来画图。

    for epoch in range(config["num_epochs"]):  # 外层循环控制训练轮数，epoch 依次是 0 到 num_epochs - 1。
        w, b, train_loss = train_one_epoch(  # 训练完一轮后返回更新后的参数和平均训练损失。
            x_train,
            y_train,
            w,
            b,
            config["batch_size"],
            config["learning_rate"],
        )
        test_metrics = evaluate(x_test, y_test, w, b, config["accuracy_threshold"])  # 在完整测试集上评估，返回指标字典。

        train_loss_history.append(train_loss)  # 把当前轮训练损失追加到列表中，列表长度逐轮增加。
        test_loss_history.append(test_metrics["loss"])  # 把当前轮测试损失追加到列表中，后面画 loss 曲线会用到。
        test_accuracy_history.append(test_metrics["threshold_accuracy"])  # 把当前轮测试 accuracy 追加到列表中，后面画 accuracy 曲线会用到。

        if epoch % 10 == 0:  # 当前训练 100 轮，所以每 10 轮打印一次日志，便于观察收敛趋势。
            print(
                f"Epoch {epoch}, "  # 打印当前是第几轮训练。
                f"Train Loss: {train_loss:.4f}, "  # 打印训练集平均损失。
                f"Test Loss: {test_metrics['loss']:.4f}, "  # 打印测试集平均损失。
                f"MAE: {test_metrics['mae']:.4f}, "  # 打印平均绝对误差。
                f"RMSE: {test_metrics['rmse']:.4f}, "  # 打印均方根误差。
                f"R2: {test_metrics['r2']:.4f}, "  # 打印拟合优度。
                f"Test Accuracy: {test_metrics['threshold_accuracy']:.4f}"  # 打印阈值型回归 accuracy。
            )

    final_metrics = evaluate(x_test, y_test, w, b, config["accuracy_threshold"])  # 最终统一输出一组完整回归指标。
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
