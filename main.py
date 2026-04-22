# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便同时写英文和中文注释。
"""Binary and multiclass logistic regression examples with scikit-learn."""  # 用一句话说明这个脚本的用途。

import os  # 导入 os，用来设置 Matplotlib 的缓存目录和后端相关环境变量。

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))  # 把 Matplotlib 缓存目录设到当前项目下，避免写入用户目录失败。

import matplotlib  # 导入 Matplotlib 主模块，用来切换无界面后端。

matplotlib.use("Agg")  # 使用无界面后端，避免在终端或沙箱环境里弹图时报图形连接错误。

import matplotlib.pyplot as plt  # 导入画图库，用来画概率分布图和混淆矩阵图。
import numpy as np  # 导入 NumPy，用来处理数组和做简单统计。
from sklearn.datasets import load_breast_cancer, load_iris  # 导入二分类和三分类常用示例数据集。
from sklearn.linear_model import LogisticRegression  # 导入 sklearn 的逻辑回归模型。
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # 导入常见分类指标函数。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分训练集和测试集。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，让特征尺度更统一，逻辑回归通常更稳。


# 以后我们每次写模型代码都默认检查这些问题，并尽量直接落实到代码结构里。
# 1. 怎么加速: 传统机器学习优先考虑 sklearn；数据量更大时再考虑并行、采样或换更强库。
# 2. 多模态吗: 当前这两个例子都是单模态表格分类任务，不是图像/文本/语音多模态任务。
# 3. 用不用 PyTorch: 这里只是经典逻辑回归，sklearn 比 torch 更直接，也更像面试常见写法。
# 4. metric 要有哪些: 分类常见看 Accuracy、Precision、Recall、F1，类别不平衡时不能只看 Accuracy。
# 5. train/test 要严格分开: 预处理只能在训练集 fit，然后再 transform 测试集，避免数据泄漏。
# 6. loss 用什么: 二分类常对应 binary cross entropy，多分类常对应 softmax + cross entropy 思路。
# 7. 参数怎么调: C、penalty、solver、class_weight、threshold 都值得考虑。
# 8. 代码里要留出问题清单: 让后续继续扩展时不漏掉指标、数据泄漏和任务定义。


def binary_metrics(y_true, y_pred):  # 二分类常见指标函数，帮助统一统计 Accuracy/Precision/Recall/F1。
    accuracy = accuracy_score(y_true, y_pred)  # 准确率，表示预测正确的样本比例，输出是 Python 浮点数。
    precision = precision_score(y_true, y_pred)  # 精确率，表示预测为正类里有多少是真的正类。
    recall = recall_score(y_true, y_pred)  # 召回率，表示真实正类里有多少被模型找出来了。
    f1 = f1_score(y_true, y_pred)  # F1 是 precision 和 recall 的调和平均，更适合综合看分类效果。
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def multiclass_metrics(y_true, y_pred):  # 多分类指标函数，这里用宏平均来公平看每一类的表现。
    accuracy = accuracy_score(y_true, y_pred)  # 准确率，表示整体预测正确的样本比例。
    precision = precision_score(y_true, y_pred, average="macro")  # 宏平均精确率，先算每类 precision 再平均。
    recall = recall_score(y_true, y_pred, average="macro")  # 宏平均召回率，先算每类 recall 再平均。
    f1 = f1_score(y_true, y_pred, average="macro")  # 宏平均 F1，类别更均衡时常用这种统计方式。
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def prepare_binary_data(test_size, random_state):  # 准备二分类数据，这里使用乳腺癌数据集。
    data = load_breast_cancer()  # 加载乳腺癌二分类数据集，里面包含特征矩阵和 0/1 标签。
    x = data.data  # 输入特征矩阵，shape = (569, 30)，每行 1 个样本、每列 1 个特征。
    y = data.target  # 二分类标签向量，shape = (569,)。

    x_train, x_test, y_train, y_test = train_test_split(
        x,  # 全部特征，shape = (569, 30)。
        y,  # 全部标签，shape = (569,)。
        test_size=test_size,  # 抽一部分作为测试集，默认这里会取 20%。
        random_state=random_state,  # 固定随机种子，保证每次 train/test 划分一致。
        stratify=y,  # 按类别比例分层抽样，保证训练集和测试集类别分布更稳定。
    )

    scaler = StandardScaler()  # 创建标准化器对象，只允许在训练集上学习均值和方差。
    x_train = scaler.fit_transform(x_train)  # 在训练集上先 fit 再 transform，shape 仍然是 (num_train_samples, 30)。
    x_test = scaler.transform(x_test)  # 用训练集的标准化规则转换测试集，shape 仍然是 (num_test_samples, 30)。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32 类型数组，shape = (num_train_samples, 30)。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32 类型数组，shape = (num_test_samples, 30)。
    y_train = y_train.astype(np.int64)  # 把训练标签转成 int64 类型数组，shape = (num_train_samples,)。
    y_test = y_test.astype(np.int64)  # 把测试标签转成 int64 类型数组，shape = (num_test_samples,)。
    return x_train, x_test, y_train, y_test, scaler, data  # 返回训练集、测试集、标准化器和原始数据对象。


def prepare_multiclass_data(test_size, random_state):  # 准备多分类数据，这里使用 Iris 三分类数据集。
    data = load_iris()  # 加载 Iris 三分类数据集，标签是 0/1/2 三个类别。
    x = data.data  # 输入特征矩阵，shape = (150, 4)，每行 1 个样本、每列 1 个特征。
    y = data.target  # 三分类标签向量，shape = (150,)。

    x_train, x_test, y_train, y_test = train_test_split(
        x,  # 全部特征，shape = (150, 4)。
        y,  # 全部标签，shape = (150,)。
        test_size=test_size,  # 抽一部分作为测试集，默认这里会取 20%。
        random_state=random_state,  # 固定随机种子，保证每次划分一致。
        stratify=y,  # 按类别比例分层抽样，保证三类在训练集和测试集里比例更稳定。
    )

    scaler = StandardScaler()  # 创建标准化器对象，只允许在训练集上学习均值和方差。
    x_train = scaler.fit_transform(x_train)  # 在训练集上先 fit 再 transform，shape 仍然是 (num_train_samples, 4)。
    x_test = scaler.transform(x_test)  # 用训练集的标准化规则转换测试集，shape 仍然是 (num_test_samples, 4)。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32 类型数组，shape = (num_train_samples, 4)。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32 类型数组，shape = (num_test_samples, 4)。
    y_train = y_train.astype(np.int64)  # 把训练标签转成 int64 类型数组，shape = (num_train_samples,)。
    y_test = y_test.astype(np.int64)  # 把测试标签转成 int64 类型数组，shape = (num_test_samples,)。
    return x_train, x_test, y_train, y_test, scaler, data  # 返回训练集、测试集、标准化器和原始数据对象。


def train_binary_model(x_train, y_train, c_value, max_iter):  # 训练二分类逻辑回归模型。
    model = LogisticRegression(
        C=c_value,  # C 是正则化强度的倒数，C 越大正则越弱，C 越小正则越强。
        max_iter=max_iter,  # 最大迭代轮数，防止优化还没收敛就提前停掉。
        solver="lbfgs",  # lbfgs 是 sklearn 逻辑回归里很常见的优化器之一。
    )
    model.fit(x_train, y_train)  # 在训练集上拟合逻辑回归模型，x_train shape = (num_train_samples, 30)，y_train shape = (num_train_samples,)。
    return model  # 返回训练好的二分类模型。


def train_multiclass_model(x_train, y_train, c_value, max_iter):  # 训练三分类逻辑回归模型。
    model = LogisticRegression(
        C=c_value,  # C 是正则化强度的倒数，C 越大正则越弱，C 越小正则越强。
        max_iter=max_iter,  # 最大迭代轮数，防止优化还没收敛就提前停掉。
        solver="lbfgs",  # lbfgs 支持多分类 softmax 逻辑回归。
    )
    model.fit(x_train, y_train)  # 在训练集上拟合三分类逻辑回归模型，x_train shape = (num_train_samples, 4)，y_train shape = (num_train_samples,)。
    return model  # 返回训练好的多分类模型。


def inference_binary(model, x_data):  # 二分类推理函数，帮助把 inference 和 evaluation 区分更清楚。
    y_pred = model.predict(x_data)  # 预测类别标签，输出 shape = (num_samples,)。
    y_prob = model.predict_proba(x_data)[:, 1]  # 预测为正类的概率，先得到 (num_samples, 2)，再取第 2 列后 shape = (num_samples,)。
    return y_pred, y_prob  # 返回预测标签和正类概率。


def inference_multiclass(model, x_data):  # 多分类推理函数，帮助你直接看到 softmax 场景下的输出。
    y_pred = model.predict(x_data)  # 预测类别标签，输出 shape = (num_samples,)。
    y_prob = model.predict_proba(x_data)  # 预测每一类的概率，shape = (num_samples, 3)，每一行三个概率和为 1。
    return y_pred, y_prob  # 返回预测标签和三类概率矩阵。


def evaluate_binary(model, x_data, y_data):  # 二分类评估函数。
    y_pred, y_prob = inference_binary(model, x_data)  # 先做推理，再拿预测结果去算指标。
    metrics = binary_metrics(y_data, y_pred)  # 基于真实标签和预测标签统一计算二分类指标。
    matrix = confusion_matrix(y_data, y_pred)  # 计算混淆矩阵，shape = (2, 2)。
    return y_pred, y_prob, metrics, matrix  # 返回预测结果、概率、指标字典和混淆矩阵。


def evaluate_multiclass(model, x_data, y_data):  # 多分类评估函数。
    y_pred, y_prob = inference_multiclass(model, x_data)  # 先做推理，再拿预测结果去算指标。
    metrics = multiclass_metrics(y_data, y_pred)  # 基于真实标签和预测标签统一计算多分类指标。
    matrix = confusion_matrix(y_data, y_pred)  # 计算混淆矩阵，shape = (3, 3)。
    return y_pred, y_prob, metrics, matrix  # 返回预测结果、概率、指标字典和混淆矩阵。


def plot_binary_results(y_true, y_prob, matrix, plot_path):  # 把二分类可视化逻辑单独拆出来。
    plt.figure(figsize=(12, 5))  # 创建一个宽 12、高 5 的画布，让两张图并排显示。

    plt.subplot(1, 2, 1)  # 选择左边第 1 张子图，画正类概率分布。
    plt.hist(y_prob[y_true == 0], bins=20, alpha=0.6, label="Class 0")  # 画真实负类样本的正类概率分布。
    plt.hist(y_prob[y_true == 1], bins=20, alpha=0.6, label="Class 1")  # 画真实正类样本的正类概率分布。
    plt.xlabel("Predicted Positive Probability")  # 设置横轴名称为预测为正类的概率。
    plt.ylabel("Count")  # 设置纵轴名称为样本数。
    plt.title("Binary Probability Distribution")  # 设置左图标题。
    plt.legend()  # 显示图例，区分两类样本。

    plt.subplot(1, 2, 2)  # 选择右边第 2 张子图，画混淆矩阵热力图。
    plt.imshow(matrix, cmap="Blues")  # 用热力图显示 2x2 混淆矩阵。
    plt.title("Binary Confusion Matrix")  # 设置右图标题。
    plt.xlabel("Predicted Label")  # 设置横轴名称为预测标签。
    plt.ylabel("True Label")  # 设置纵轴名称为真实标签。
    for row in range(matrix.shape[0]):  # 遍历混淆矩阵的每一行。
        for col in range(matrix.shape[1]):  # 遍历混淆矩阵的每一列。
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="black")  # 在格子中间写入具体计数值。

    plt.tight_layout()  # 自动调整子图间距，避免标题和坐标轴文字重叠。
    plt.savefig(plot_path, dpi=200)  # 把图直接保存成图片文件，避免 plt.show() 依赖图形界面。


def plot_multiclass_results(matrix, plot_path):  # 把多分类可视化逻辑单独拆出来。
    plt.figure(figsize=(6, 5))  # 创建一个更紧凑的画布，用来显示 3x3 混淆矩阵。
    plt.imshow(matrix, cmap="Greens")  # 用热力图显示 3x3 混淆矩阵。
    plt.title("Multiclass Confusion Matrix")  # 设置图标题。
    plt.xlabel("Predicted Label")  # 设置横轴名称为预测标签。
    plt.ylabel("True Label")  # 设置纵轴名称为真实标签。
    for row in range(matrix.shape[0]):  # 遍历混淆矩阵的每一行。
        for col in range(matrix.shape[1]):  # 遍历混淆矩阵的每一列。
            plt.text(col, row, matrix[row, col], ha="center", va="center", color="black")  # 在格子中间写入具体计数值。
    plt.tight_layout()  # 自动调整图像布局，避免标题和坐标轴文字重叠。
    plt.savefig(plot_path, dpi=200)  # 把图直接保存成图片文件。


def main():  # 主入口函数，把二分类和多分类两个例子都串起来。
    config = {
        "test_size": 0.2,  # 测试集占比，表示 20% 的样本留给最终评估。
        "random_state": 42,  # 固定随机种子，保证切分结果稳定，方便复现实验。
        "c_value": 1.0,  # 正则化强度倒数 C，越大正则越弱，越小正则越强。
        "max_iter": 1000,  # 最大迭代轮数，如果不够大可能会出现逻辑回归还没收敛。
        "binary_plot_path": "binary_logistic_regression.png",  # 二分类概率分布图和混淆矩阵图的保存路径。
        "multiclass_plot_path": "multiclass_logistic_regression.png",  # 多分类混淆矩阵图的保存路径。
    }
    # 这份 config 是后面面试或快速实验时最先看的地方。
    # 逻辑回归最常见的追问点:
    # 1. 为什么二分类常讲 sigmoid，多分类常讲 softmax
    # 2. 为什么要做标准化
    # 3. C 调大调小各意味着什么
    # 4. predict 和 predict_proba 有什么区别

    binary_x_train, binary_x_test, binary_y_train, binary_y_test, binary_scaler, binary_data = prepare_binary_data(  # 准备二分类训练集和测试集。
        config["test_size"],
        config["random_state"],
    )
    binary_model = train_binary_model(binary_x_train, binary_y_train, config["c_value"], config["max_iter"])  # 训练二分类逻辑回归模型。
    binary_train_pred, binary_train_prob, binary_train_metrics, binary_train_matrix = evaluate_binary(  # 在训练集上评估二分类模型。
        binary_model,
        binary_x_train,
        binary_y_train,
    )
    binary_test_pred, binary_test_prob, binary_test_metrics, binary_test_matrix = evaluate_binary(  # 在测试集上评估二分类模型。
        binary_model,
        binary_x_test,
        binary_y_test,
    )

    multiclass_x_train, multiclass_x_test, multiclass_y_train, multiclass_y_test, multiclass_scaler, multiclass_data = prepare_multiclass_data(  # 准备三分类训练集和测试集。
        config["test_size"],
        config["random_state"],
    )
    multiclass_model = train_multiclass_model(multiclass_x_train, multiclass_y_train, config["c_value"], config["max_iter"])  # 训练多分类逻辑回归模型。
    multiclass_train_pred, multiclass_train_prob, multiclass_train_metrics, multiclass_train_matrix = evaluate_multiclass(  # 在训练集上评估三分类模型。
        multiclass_model,
        multiclass_x_train,
        multiclass_y_train,
    )
    multiclass_test_pred, multiclass_test_prob, multiclass_test_metrics, multiclass_test_matrix = evaluate_multiclass(  # 在测试集上评估三分类模型。
        multiclass_model,
        multiclass_x_test,
        multiclass_y_test,
    )

    print("Binary Classification Example")  # 打印二分类例子标题。
    print(f"Binary feature matrix shape: {binary_data.data.shape}")  # 打印二分类原始特征矩阵 shape，当前一般是 (569, 30)。
    print(f"Binary scaler mean shape: {binary_scaler.mean_.shape}")  # 打印二分类标准化器均值向量 shape，当前一般是 (30,)。
    print(f"Binary model coef shape: {binary_model.coef_.shape}")  # 打印二分类权重 shape，通常是 (1, 30)。
    print(f"Binary model intercept shape: {binary_model.intercept_.shape}")  # 打印二分类偏置 shape，通常是 (1,)。
    print(f"Binary train probability shape: {binary_train_prob.shape}")  # 打印二分类训练集正类概率向量 shape，通常是 (num_train_samples,)。
    print(f"Binary test probability shape: {binary_test_prob.shape}")  # 打印二分类测试集正类概率向量 shape，通常是 (num_test_samples,)。
    print("Binary Train Accuracy:", binary_train_metrics["accuracy"])  # 打印二分类训练集准确率。
    print("Binary Train Precision:", binary_train_metrics["precision"])  # 打印二分类训练集精确率。
    print("Binary Train Recall:", binary_train_metrics["recall"])  # 打印二分类训练集召回率。
    print("Binary Train F1:", binary_train_metrics["f1"])  # 打印二分类训练集 F1。
    print("Binary Test Accuracy:", binary_test_metrics["accuracy"])  # 打印二分类测试集准确率。
    print("Binary Test Precision:", binary_test_metrics["precision"])  # 打印二分类测试集精确率。
    print("Binary Test Recall:", binary_test_metrics["recall"])  # 打印二分类测试集召回率。
    print("Binary Test F1:", binary_test_metrics["f1"])  # 打印二分类测试集 F1。

    print("\nMulticlass Classification Example")  # 打印多分类例子标题。
    print(f"Multiclass feature matrix shape: {multiclass_data.data.shape}")  # 打印多分类原始特征矩阵 shape，当前一般是 (150, 4)。
    print(f"Multiclass scaler mean shape: {multiclass_scaler.mean_.shape}")  # 打印多分类标准化器均值向量 shape，当前一般是 (4,)。
    print(f"Multiclass model coef shape: {multiclass_model.coef_.shape}")  # 打印多分类权重 shape，当前通常是 (3, 4)。
    print(f"Multiclass model intercept shape: {multiclass_model.intercept_.shape}")  # 打印多分类偏置 shape，当前通常是 (3,)。
    print(f"Multiclass train probability shape: {multiclass_train_prob.shape}")  # 打印多分类训练集概率矩阵 shape，通常是 (num_train_samples, 3)。
    print(f"Multiclass test probability shape: {multiclass_test_prob.shape}")  # 打印多分类测试集概率矩阵 shape，通常是 (num_test_samples, 3)。
    print("Multiclass Train Accuracy:", multiclass_train_metrics["accuracy"])  # 打印多分类训练集准确率。
    print("Multiclass Train Precision Macro:", multiclass_train_metrics["precision_macro"])  # 打印多分类训练集宏平均精确率。
    print("Multiclass Train Recall Macro:", multiclass_train_metrics["recall_macro"])  # 打印多分类训练集宏平均召回率。
    print("Multiclass Train F1 Macro:", multiclass_train_metrics["f1_macro"])  # 打印多分类训练集宏平均 F1。
    print("Multiclass Test Accuracy:", multiclass_test_metrics["accuracy"])  # 打印多分类测试集准确率。
    print("Multiclass Test Precision Macro:", multiclass_test_metrics["precision_macro"])  # 打印多分类测试集宏平均精确率。
    print("Multiclass Test Recall Macro:", multiclass_test_metrics["recall_macro"])  # 打印多分类测试集宏平均召回率。
    print("Multiclass Test F1 Macro:", multiclass_test_metrics["f1_macro"])  # 打印多分类测试集宏平均 F1。
    # 这里最值得观察的区别是:
    # 1. 二分类 predict_proba 输出 shape = (num_samples, 2)，但我们常只取正类那一列。
    # 2. 多分类 predict_proba 输出 shape = (num_samples, 3)，每一行三个概率和等于 1，更贴近 softmax 场景。

    plot_binary_results(binary_y_test, binary_test_prob, binary_test_matrix, config["binary_plot_path"])  # 画二分类概率分布图和混淆矩阵图。
    plot_multiclass_results(multiclass_test_matrix, config["multiclass_plot_path"])  # 画多分类混淆矩阵图。
    print(f"Saved binary plot to: {config['binary_plot_path']}")  # 打印二分类图片保存路径。
    print(f"Saved multiclass plot to: {config['multiclass_plot_path']}")  # 打印多分类图片保存路径。


if __name__ == "__main__":  # 直接运行这个文件时才执行 main，被别的文件导入时不会自动训练。
    main()
