# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Simple linear regression interview template with scikit-learn."""  # 用一句话说明这个脚本的用途。

import os  # 导入 os，用来设置 Matplotlib 的缓存目录和后端相关环境变量。

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))  # 把 Matplotlib 缓存目录设到当前项目下，避免写入用户目录失败。

import matplotlib  # 导入 Matplotlib 主模块，用来切换无界面后端。

matplotlib.use("Agg")  # 使用无界面后端，避免在终端或沙箱环境里弹图时报图形连接错误。

import matplotlib.pyplot as plt  # 导入画图库，用来画预测散点图和误差图。
import numpy as np  # 导入 NumPy，用来处理数组和做一些简单统计。
from sklearn.datasets import fetch_california_housing  # 导入加州房价数据集，作为线性回归经典表格例子。
from sklearn.linear_model import LinearRegression  # 导入 sklearn 的线性回归模型，面试里最常见的版本之一。
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 导入常见回归指标函数。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分训练集和测试集。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，让特征更适合训练和比较。


# 以后我们每次写模型代码都默认检查这些问题，并尽量直接落实到代码结构里。
# 1. 怎么加速: 传统机器学习优先考虑 sklearn；数据量更大时再考虑并行、采样或换更强库。
# 2. 多模态吗: 当前这个例子是单模态表格回归，不是图像/文本/语音多模态任务。
# 3. 用不用 PyTorch: 这里只是经典线性回归，sklearn 比 torch 更直接，也更像面试常见写法。
# 4. metric 要有哪些: 回归常见看 MSE、MAE、RMSE、R2，不只盯着一个指标。
# 5. train/test 要严格分开: 预处理只能在训练集 fit，然后再 transform 测试集，避免数据泄漏。
# 6. loss 用什么: sklearn 的 LinearRegression 默认走最小二乘思路，本质上就是在优化平方误差。
# 7. 参数怎么调: 线性回归本身超参数不多，重点更多在特征工程、标准化、异常值和模型选择。
# 8. 代码里要留出问题清单: 让后续继续扩展时不漏掉指标、数据泄漏和任务定义。


def regression_metrics(pred, target, threshold):  # 统一计算多个回归指标，避免只盯着单一 loss。
    abs_error = np.abs(pred - target)  # 逐样本绝对误差，shape = (num_samples,)。
    mse = mean_squared_error(target, pred)  # 均方误差，输出是 Python 浮点数，没有数组 shape。
    mae = mean_absolute_error(target, pred)  # 平均绝对误差，输出是 Python 浮点数。
    rmse = np.sqrt(mse)  # 对 MSE 开平方得到 RMSE，输出也是 Python 浮点数。
    accuracy = np.mean(abs_error < threshold)  # 自定义“回归 accuracy”，误差小于阈值记为正确。
    r2 = r2_score(target, pred)  # 计算 R2，帮助判断模型解释目标方差的能力。
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "threshold_accuracy": float(accuracy),
        "r2": float(r2),
        "loss": float(mse),  # 这里把 MSE 同时当作 loss，方便后面统一打印和取值。
    }


def prepare_data(test_size, random_state):  # 负责数据下载、切分、标准化和类型转换。
    data = fetch_california_housing()  # 下载或读取加州房价数据集，里面包含特征矩阵和回归标签。
    x = data.data  # 取出输入特征矩阵，shape = (20640, 8)，每行 1 个样本、每列 1 个特征。
    y = data.target  # 取出每个样本对应的房价标签，shape = (20640,)。

    x_train, x_test, y_train, y_test = train_test_split(
        x,  # 原始全部特征，shape = (20640, 8)。
        y,  # 原始全部标签，shape = (20640,)。
        test_size=test_size,  # 抽出一部分作为测试集，默认这里会取 20%。
        random_state=random_state,  # 固定随机种子，保证每次 train/test 划分一致。
    )

    scaler = StandardScaler()  # 创建标准化器对象，只允许在训练集上学习均值和方差。
    x_train = scaler.fit_transform(x_train)  # 在训练集上先拟合再标准化，shape 仍然是 (num_train_samples, 8)。
    x_test = scaler.transform(x_test)  # 用训练集得到的标准化规则去转换测试集，shape 仍然是 (num_test_samples, 8)。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32 类型数组，shape = (num_train_samples, 8)。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32 类型数组，shape = (num_test_samples, 8)。
    y_train = y_train.astype(np.float32)  # 把训练标签转成 float32 类型数组，shape = (num_train_samples,)。
    y_test = y_test.astype(np.float32)  # 把测试标签转成 float32 类型数组，shape = (num_test_samples,)。
    return x_train, x_test, y_train, y_test, scaler  # 返回训练集、测试集和标准化器，后面推理时还能复用。


def train_model(x_train, y_train):  # 单独封装训练逻辑，让主函数更清楚。
    model = LinearRegression()  # 创建 sklearn 线性回归模型对象，面试里最常见的写法就是这一句。
    model.fit(x_train, y_train)  # 在训练集上拟合模型，x_train shape = (num_train_samples, 8)，y_train shape = (num_train_samples,)。
    return model  # 返回训练好的模型对象，后面可直接 predict。


def evaluate(model, x_data, y_data, accuracy_threshold):  # 单独封装评估逻辑，让 train/test 指标计算风格统一。
    pred = model.predict(x_data)  # 用训练好的模型做预测，x_data shape = (num_samples, 8)，pred shape = (num_samples,)。
    metrics = regression_metrics(pred, y_data, accuracy_threshold)  # 基于整份数据统一计算回归指标。
    return pred, metrics  # 返回预测结果和指标字典，后面画图和打印都方便。


def plot_results(y_test, pred_test, plot_path):  # 把可视化逻辑单独拆出来，主流程更干净。
    plt.figure(figsize=(12, 5))  # 创建一个宽 12、高 5 的画布，让两张图并排显示。

    plt.subplot(1, 2, 1)  # 选择左边第 1 张子图，画真实值和预测值的散点图。
    plt.scatter(y_test, pred_test, alpha=0.4)  # 横轴是真实值，纵轴是预测值，点越贴近对角线说明预测越准。
    min_value = min(y_test.min(), pred_test.min())  # 取横纵坐标共同的最小值，用来画参考对角线。
    max_value = max(y_test.max(), pred_test.max())  # 取横纵坐标共同的最大值，用来画参考对角线。
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")  # 画一条理想预测线 y = x。
    plt.xlabel("True Value")  # 设置横轴名称为真实值。
    plt.ylabel("Predicted Value")  # 设置纵轴名称为预测值。
    plt.title("Prediction vs Truth")  # 设置左图标题。

    plt.subplot(1, 2, 2)  # 选择右边第 2 张子图，画误差分布图。
    residual = pred_test - y_test  # 计算预测误差，shape = (num_test_samples,)。
    plt.hist(residual, bins=30, color="orange", alpha=0.8)  # 画误差直方图，看误差是否集中在 0 附近。
    plt.xlabel("Residual")  # 设置横轴名称为误差。
    plt.ylabel("Count")  # 设置纵轴名称为样本数。
    plt.title("Residual Distribution")  # 设置右图标题。

    plt.tight_layout()  # 自动调整子图间距，避免标题和坐标轴文字重叠。
    plt.savefig(plot_path, dpi=200)  # 把图直接保存成图片文件，避免 plt.show() 依赖图形界面。


def main():  # 主入口函数，后面你继续扩展模型时就改这里和对应子函数。
    config = {
        "test_size": 0.2,  # 测试集占比，表示 20% 的样本留给最终评估。
        "random_state": 42,  # 固定随机种子，保证切分结果稳定，方便复现实验。
        "accuracy_threshold": 0.5,  # 自定义“回归 accuracy”阈值，误差小于 0.5 就算预测正确。
        "plot_path": "sklearn_linear_regression.png",  # 预测散点图和误差图的保存路径。
    }
    # 这份 config 是后面面试或快速实验时最先看的地方。
    # sklearn 版线性回归超参数不多，所以真正更重要的是:
    # 1. 数据预处理对不对
    # 2. train/test 有没有分干净
    # 3. metric 是否完整
    # 4. 什么时候线性回归不够用，要换更强模型

    x_train, x_test, y_train, y_test, scaler = prepare_data(config["test_size"], config["random_state"])  # 准备训练集和测试集，并保证预处理只在训练集 fit。
    model = train_model(x_train, y_train)  # 训练 sklearn 线性回归模型。

    train_pred, train_metrics = evaluate(model, x_train, y_train, config["accuracy_threshold"])  # 在训练集上评估，帮助判断是否欠拟合或过拟合。
    test_pred, test_metrics = evaluate(model, x_test, y_test, config["accuracy_threshold"])  # 在测试集上评估，帮助判断泛化能力。

    print(f"Scaler mean shape: {scaler.mean_.shape}")  # 打印标准化器均值向量的 shape，当前一般是 (8,)。
    print(f"Model coef shape: {model.coef_.shape}")  # 打印模型权重 shape，当前一般是 (8,)。
    print(f"Model intercept shape: scalar")  # 打印模型偏置是标量，不是向量。

    print("\nTrain Metrics")  # 打印训练集指标标题。
    print("Train Loss:", train_metrics["loss"])  # 打印训练集 MSE loss。
    print("Train MAE:", train_metrics["mae"])  # 打印训练集平均绝对误差。
    print("Train RMSE:", train_metrics["rmse"])  # 打印训练集均方根误差。
    print("Train R2:", train_metrics["r2"])  # 打印训练集 R2。
    print("Train Accuracy:", train_metrics["threshold_accuracy"])  # 打印训练集阈值型回归 accuracy。

    print("\nTest Metrics")  # 打印测试集指标标题。
    print("Test Loss:", test_metrics["loss"])  # 打印测试集 MSE loss。
    print("Test MAE:", test_metrics["mae"])  # 打印测试集平均绝对误差。
    print("Test RMSE:", test_metrics["rmse"])  # 打印测试集均方根误差。
    print("Test R2:", test_metrics["r2"])  # 打印测试集 R2。
    print("Test Accuracy:", test_metrics["threshold_accuracy"])  # 打印测试集阈值型回归 accuracy。
    # 看结果时不要只盯着一个指标。
    # 回归任务通常至少同时看 loss、MAE、RMSE、R2。
    # 如果 train 指标很好但 test 指标差，通常要怀疑过拟合、数据泄漏或训练/测试分布差异。

    plot_results(y_test, test_pred, config["plot_path"])  # 根据测试集真实值和预测值画图，方便直观看效果。
    print(f"Saved plot to: {config['plot_path']}")  # 打印图片保存路径，方便确认脚本完整执行结束。


if __name__ == "__main__":  # 直接运行这个文件时才执行 main，被别的文件导入时不会自动训练。
    main()
