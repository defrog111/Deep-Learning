# -*- coding: utf-8 -*-  # 声明文件使用 UTF-8 编码，方便整份脚本写中文注释。
"""Detailed NumPy logistic regression interview example."""  # 用一句话说明这个脚本的用途。

import numpy as np  # 导入 NumPy，用来做矩阵乘法、sigmoid、loss 和梯度下降。
from sklearn.datasets import load_breast_cancer  # 导入乳腺癌二分类数据集，作为手写 logistic regression 示例。
from sklearn.model_selection import train_test_split  # 导入数据切分工具，用于划分 train、val、test。
from sklearn.preprocessing import StandardScaler  # 导入标准化工具，只能在训练集上 fit。


def prepare_data(random_state=42):  # 负责读取数据、切分数据、标准化数据和类型转换。
    dataset = load_breast_cancer()  # 读取乳腺癌数据集对象，这里不是张量，没有 shape。
    x = dataset.data  # 取出特征矩阵，shape = (569, 30)。
    y = dataset.target  # 取出二分类标签向量，shape = (569,)。

    x_train_full, x_test, y_train_full, y_test = train_test_split(  # 先切出训练+验证集合和测试集合。
        x,  # 传入全部特征矩阵，shape = (569, 30)。
        y,  # 传入全部标签向量，shape = (569,)。
        test_size=0.2,  # 取 20% 作为测试集，这里不是张量，没有 shape。
        random_state=random_state,  # 固定随机种子，保证切分结果稳定，这里不是张量，没有 shape。
        stratify=y,  # 按类别比例分层抽样，保证 0/1 标签分布稳定，这里不是张量，没有 shape。
    )  # 切分后 x_train_full shape 约是 (455, 30)，x_test shape 约是 (114, 30)。

    x_train, x_val, y_train, y_val = train_test_split(  # 再从训练+验证集合里切出训练集和验证集。
        x_train_full,  # 传入训练+验证特征矩阵，shape 约是 (455, 30)。
        y_train_full,  # 传入训练+验证标签向量，shape 约是 (455,)。
        test_size=0.2,  # 再取 20% 作为验证集，这里不是张量，没有 shape。
        random_state=random_state,  # 固定随机种子，保证切分结果稳定，这里不是张量，没有 shape。
        stratify=y_train_full,  # 继续按类别比例分层抽样，保证验证集类别稳定，这里不是张量，没有 shape。
    )  # 切分后 x_train shape 约是 (364, 30)，x_val shape 约是 (91, 30)。

    scaler = StandardScaler()  # 创建标准化器对象，这里不是张量，没有 shape。
    x_train = scaler.fit_transform(x_train)  # 在训练集上 fit 并 transform，shape 仍然是 (364, 30) 左右。
    x_val = scaler.transform(x_val)  # 用训练集统计量标准化验证集，shape 仍然是 (91, 30) 左右。
    x_test = scaler.transform(x_test)  # 用训练集统计量标准化测试集，shape 仍然是 (114, 30) 左右。

    x_train = x_train.astype(np.float32)  # 把训练特征转成 float32 数组，shape = (364, 30) 左右。
    x_val = x_val.astype(np.float32)  # 把验证特征转成 float32 数组，shape = (91, 30) 左右。
    x_test = x_test.astype(np.float32)  # 把测试特征转成 float32 数组，shape = (114, 30) 左右。
    y_train = y_train.astype(np.float32)  # 把训练标签转成 float32 数组，shape = (364,) 左右。
    y_val = y_val.astype(np.float32)  # 把验证标签转成 float32 数组，shape = (91,) 左右。
    y_test = y_test.astype(np.float32)  # 把测试标签转成 float32 数组，shape = (114,) 左右。
    return x_train, x_val, x_test, y_train, y_val, y_test, scaler, dataset  # 返回 train、val、test、标准化器和数据集对象。


def sigmoid(z):  # 定义 sigmoid 函数，把线性输出压到 0 到 1 之间。
    return 1.0 / (1.0 + np.exp(-z))  # 输入 z shape 常见是 (num_samples,)；输出概率 shape 仍然是 (num_samples,)。


def binary_cross_entropy(prob, target):  # 定义二分类交叉熵损失函数，输入是概率和 0/1 标签。
    eps = 1e-7  # 加一个很小的常数，避免 log(0) 导致数值溢出，这里不是张量，没有 shape。
    clipped_prob = np.clip(prob, eps, 1.0 - eps)  # 把概率裁剪到 (eps, 1-eps) 区间，shape = (num_samples,)。
    loss = -np.mean(target * np.log(clipped_prob) + (1.0 - target) * np.log(1.0 - clipped_prob))  # 计算 BCE 平均损失，输出是标量，没有数组 shape。
    return float(loss)  # 返回 Python 浮点数，方便后面打印和存历史曲线。


def classification_metrics(prob, target, threshold=0.5):  # 统一计算二分类常见指标，输入是概率和 0/1 标签。
    pred = (prob >= threshold).astype(np.int32)  # 按阈值把概率转成预测类别标签，shape = (num_samples,)。
    target_int = target.astype(np.int32)  # 把真实标签显式转成整数标签，shape = (num_samples,)。
    accuracy = np.mean(pred == target_int)  # 计算准确率，输出是标量，没有数组 shape。
    tp = np.sum((pred == 1) & (target_int == 1))  # 计算真正例个数，输出是标量，没有数组 shape。
    fp = np.sum((pred == 1) & (target_int == 0))  # 计算假正例个数，输出是标量，没有数组 shape。
    fn = np.sum((pred == 0) & (target_int == 1))  # 计算假负例个数，输出是标量，没有数组 shape。
    precision = tp / (tp + fp + 1e-7)  # 计算精确率，输出是标量，没有数组 shape。
    recall = tp / (tp + fn + 1e-7)  # 计算召回率，输出是标量，没有数组 shape。
    f1 = 2.0 * precision * recall / (precision + recall + 1e-7)  # 计算 F1，输出是标量，没有数组 shape。
    loss = binary_cross_entropy(prob, target)  # 计算当前数据集的 BCE loss，输出是标量，没有数组 shape。
    return {  # 返回一个字典，里面每个值都是 Python 浮点数或整数。
        "loss": float(loss),  # 记录 BCE loss。
        "accuracy": float(accuracy),  # 记录准确率。
        "precision": float(precision),  # 记录精确率。
        "recall": float(recall),  # 记录召回率。
        "f1": float(f1),  # 记录 F1。
        "tp": int(tp),  # 记录真正例个数。
        "fp": int(fp),  # 记录假正例个数。
        "fn": int(fn),  # 记录假负例个数。
    }


def train_logistic_regression(x_train, y_train, x_val, y_val, learning_rate=0.05, epochs=500):  # 手写 full-batch logistic regression 训练函数。
    w = np.zeros(x_train.shape[1], dtype=np.float32)  # 初始化权重向量，shape = (30,)。
    b = np.float32(0.0)  # 初始化偏置标量，这里不是数组，没有 shape。
    n = len(x_train)  # 训练样本数，是整数，没有 shape。
    train_loss_history = []  # 保存每一轮训练集 loss，列表长度最后会等于 epochs。
    val_loss_history = []  # 保存每一轮验证集 loss，列表长度最后会等于 epochs。
    train_accuracy_history = []  # 保存每一轮训练集 accuracy，列表长度最后会等于 epochs。
    val_accuracy_history = []  # 保存每一轮验证集 accuracy，列表长度最后会等于 epochs。

    for epoch in range(epochs):  # 外层循环控制训练轮数，epoch 是整数，没有 shape。
        logits = x_train @ w + b  # 线性部分 z = XW + b，shape = (num_train,)。
        prob = sigmoid(logits)  # 把线性输出变成概率，shape = (num_train,)。
        error = prob - y_train  # 计算梯度里会用到的概率误差，shape = (num_train,)。

        grad_w = (x_train.T @ error) / n  # 计算 loss 对权重的梯度，shape = (30,)。
        grad_b = np.sum(error) / n  # 计算 loss 对偏置的梯度，输出是标量，没有数组 shape。

        w -= learning_rate * grad_w  # 按梯度下降公式更新权重，shape 仍然是 (30,)。
        b -= learning_rate * grad_b  # 按梯度下降公式更新偏置，这里不是数组，没有 shape。

        train_prob = sigmoid(x_train @ w + b)  # 用更新后的参数算训练集概率，shape = (num_train,)。
        val_prob = sigmoid(x_val @ w + b)  # 用更新后的参数算验证集概率，shape = (num_val,)。
        train_metrics = classification_metrics(train_prob, y_train)  # 计算训练集指标，返回字典，没有数组 shape。
        val_metrics = classification_metrics(val_prob, y_val)  # 计算验证集指标，返回字典，没有数组 shape。

        train_loss_history.append(train_metrics["loss"])  # 记录训练集 loss，列表长度加 1。
        val_loss_history.append(val_metrics["loss"])  # 记录验证集 loss，列表长度加 1。
        train_accuracy_history.append(train_metrics["accuracy"])  # 记录训练集 accuracy，列表长度加 1。
        val_accuracy_history.append(val_metrics["accuracy"])  # 记录验证集 accuracy，列表长度加 1。

        if epoch % 50 == 0 or epoch == epochs - 1:  # 每 50 轮和最后一轮打印一次日志，这里不是张量，没有 shape。
            print(  # 打印当前轮训练日志，这里不是张量，没有 shape。
                f"Epoch {epoch:03d} | "  # 打印轮数，这是字符串，没有 shape。
                f"Train Loss: {train_metrics['loss']:.4f} | "  # 打印训练集 BCE loss，这是字符串，没有 shape。
                f"Val Loss: {val_metrics['loss']:.4f} | "  # 打印验证集 BCE loss，这是字符串，没有 shape。
                f"Train Acc: {train_metrics['accuracy']:.4f} | "  # 打印训练集 accuracy，这是字符串，没有 shape。
                f"Val Acc: {val_metrics['accuracy']:.4f}"  # 打印验证集 accuracy，这是字符串，没有 shape。
            )

    history = {  # 把训练过程历史收进字典，后面统一打印和扩展。
        "train_loss": train_loss_history,  # 训练集 loss 历史，列表长度等于 epochs。
        "val_loss": val_loss_history,  # 验证集 loss 历史，列表长度等于 epochs。
        "train_accuracy": train_accuracy_history,  # 训练集 accuracy 历史，列表长度等于 epochs。
        "val_accuracy": val_accuracy_history,  # 验证集 accuracy 历史，列表长度等于 epochs。
    }
    return w, b, history  # 返回训练好的参数和历史记录。


def predict_probability(x_data, w, b):  # 单独封装概率预测逻辑，让主流程更清楚。
    logits = x_data @ w + b  # 先算线性输出，shape = (num_samples,)。
    prob = sigmoid(logits)  # 再经过 sigmoid 得到概率，shape = (num_samples,)。
    return prob  # 返回概率向量，后面可以继续算 loss 或阈值化。


def main():  # 主入口函数，把完整流程串起来。
    config = {  # 把常改超参数统一放在这里，后面调参更方便。
        "learning_rate": 0.05,  # 学习率，控制每次梯度下降更新步长，这里不是张量，没有 shape。
        "epochs": 500,  # 训练轮数，控制完整遍历训练集的次数，这里不是张量，没有 shape。
        "threshold": 0.5,  # 二分类概率阈值，大于等于 0.5 判成 1，这里不是张量，没有 shape。
    }

    x_train, x_val, x_test, y_train, y_val, y_test, scaler, dataset = prepare_data()  # 准备 train、val、test 三份数据。
    w, b, history = train_logistic_regression(  # 训练手写 NumPy 版 logistic regression。
        x_train,  # 训练特征矩阵，shape 约是 (364, 30)。
        y_train,  # 训练标签向量，shape 约是 (364,)。
        x_val,  # 验证特征矩阵，shape 约是 (91, 30)。
        y_val,  # 验证标签向量，shape 约是 (91,)。
        learning_rate=config["learning_rate"],  # 传入学习率，这里不是张量，没有 shape。
        epochs=config["epochs"],  # 传入训练轮数，这里不是张量，没有 shape。
    )

    train_prob = predict_probability(x_train, w, b)  # 对训练集做最终概率预测，shape = (num_train,)。
    val_prob = predict_probability(x_val, w, b)  # 对验证集做最终概率预测，shape = (num_val,)。
    test_prob = predict_probability(x_test, w, b)  # 对测试集做最终概率预测，shape = (num_test,)。

    train_metrics = classification_metrics(train_prob, y_train, threshold=config["threshold"])  # 计算训练集最终指标，返回字典，没有数组 shape。
    val_metrics = classification_metrics(val_prob, y_val, threshold=config["threshold"])  # 计算验证集最终指标，返回字典，没有数组 shape。
    test_metrics = classification_metrics(test_prob, y_test, threshold=config["threshold"])  # 计算测试集最终指标，返回字典，没有数组 shape。

    print("\nFeature names:")  # 打印特征名标题，这是字符串，没有 shape。
    print(dataset.feature_names)  # 打印 30 个特征名组成的列表，列表长度是 30，不是张量，没有 shape。
    print(f"\nScaler mean shape: {scaler.mean_.shape}")  # 打印标准化器均值向量 shape，当前一般是 (30,)。
    print(f"Weight shape: {w.shape}")  # 打印权重向量 shape，当前一般是 (30,)。
    print("Bias shape: scalar")  # 打印偏置是标量，不是数组，没有 shape。
    print(f"Train probability shape: {train_prob.shape}")  # 打印训练集概率向量 shape，当前一般是 (364,) 左右。
    print(f"Val probability shape: {val_prob.shape}")  # 打印验证集概率向量 shape，当前一般是 (91,) 左右。
    print(f"Test probability shape: {test_prob.shape}")  # 打印测试集概率向量 shape，当前一般是 (114,) 左右。

    print("\nFinal train metrics:")  # 打印训练集指标标题，这是字符串，没有 shape。
    print("Train Loss:", train_metrics["loss"])  # 打印训练集 BCE loss，这是字符串和浮点数组合，没有 shape。
    print("Train Accuracy:", train_metrics["accuracy"])  # 打印训练集 accuracy，这是字符串和浮点数组合，没有 shape。
    print("Train Precision:", train_metrics["precision"])  # 打印训练集 precision，这是字符串和浮点数组合，没有 shape。
    print("Train Recall:", train_metrics["recall"])  # 打印训练集 recall，这是字符串和浮点数组合，没有 shape。
    print("Train F1:", train_metrics["f1"])  # 打印训练集 F1，这是字符串和浮点数组合，没有 shape。

    print("\nFinal val metrics:")  # 打印验证集指标标题，这是字符串，没有 shape。
    print("Val Loss:", val_metrics["loss"])  # 打印验证集 BCE loss，这是字符串和浮点数组合，没有 shape。
    print("Val Accuracy:", val_metrics["accuracy"])  # 打印验证集 accuracy，这是字符串和浮点数组合，没有 shape。
    print("Val Precision:", val_metrics["precision"])  # 打印验证集 precision，这是字符串和浮点数组合，没有 shape。
    print("Val Recall:", val_metrics["recall"])  # 打印验证集 recall，这是字符串和浮点数组合，没有 shape。
    print("Val F1:", val_metrics["f1"])  # 打印验证集 F1，这是字符串和浮点数组合，没有 shape。

    print("\nFinal test metrics:")  # 打印测试集指标标题，这是字符串，没有 shape。
    print("Test Loss:", test_metrics["loss"])  # 打印测试集 BCE loss，这是字符串和浮点数组合，没有 shape。
    print("Test Accuracy:", test_metrics["accuracy"])  # 打印测试集 accuracy，这是字符串和浮点数组合，没有 shape。
    print("Test Precision:", test_metrics["precision"])  # 打印测试集 precision，这是字符串和浮点数组合，没有 shape。
    print("Test Recall:", test_metrics["recall"])  # 打印测试集 recall，这是字符串和浮点数组合，没有 shape。
    print("Test F1:", test_metrics["f1"])  # 打印测试集 F1，这是字符串和浮点数组合，没有 shape。
    print("Test TP:", test_metrics["tp"])  # 打印测试集真正例个数，这是字符串和整数组合，没有 shape。
    print("Test FP:", test_metrics["fp"])  # 打印测试集假正例个数，这是字符串和整数组合，没有 shape。
    print("Test FN:", test_metrics["fn"])  # 打印测试集假负例个数，这是字符串和整数组合，没有 shape。

    print("\nHistory length:")  # 打印历史记录标题，这是字符串，没有 shape。
    print("Train loss history length:", len(history["train_loss"]))  # 打印训练集 loss 历史长度，这是整数，没有 shape。
    print("Val loss history length:", len(history["val_loss"]))  # 打印验证集 loss 历史长度，这是整数，没有 shape。
    print("Train accuracy history length:", len(history["train_accuracy"]))  # 打印训练集 accuracy 历史长度，这是整数，没有 shape。
    print("Val accuracy history length:", len(history["val_accuracy"]))  # 打印验证集 accuracy 历史长度，这是整数，没有 shape。


if __name__ == "__main__":  # 只有直接运行这个文件时才执行 main，被导入时不会自动运行。
    main()  # 调用主函数开始完整流程，这里不是张量，没有 shape。
