# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: Gaussian Naive Bayes.\"\"\"  # 说明这道题是手写高斯朴素贝叶斯。

import numpy as np  # 导入 NumPy，用来统计均值方差和计算高斯概率。

def gaussian_log_prob(x, mean, var):  # 定义高斯分布对数概率函数，输入向量 shape = (num_features,)。
    return -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var)  # 返回当前类别下的对数概率，输出是标量。

x_train = np.array([[1.0, 2.0], [1.2, 1.8], [3.0, 3.5], [3.2, 3.8]], dtype=np.float32)  # 定义训练特征矩阵，shape = (4, 2)。
y_train = np.array([0, 0, 1, 1], dtype=np.int32)  # 定义训练标签向量，shape = (4,)。
x_test = np.array([1.1, 1.9], dtype=np.float32)  # 定义测试样本向量，shape = (2,)。
classes = np.unique(y_train)  # 取出所有类别编号，shape = (2,)。
log_probs = []  # 创建空列表保存每个类别的对数概率。

for cls in classes:  # 遍历每个类别。
    cls_x = x_train[y_train == cls]  # 取当前类别样本矩阵，shape = (num_cls_samples, 2)。
    mean = np.mean(cls_x, axis=0)  # 计算当前类别的均值向量，shape = (2,)。
    var = np.var(cls_x, axis=0) + 1e-6  # 计算当前类别的方差向量并加稳定项，shape = (2,)。
    prior = np.log(len(cls_x) / len(x_train))  # 计算当前类别的先验对数概率，输出是标量。
    log_prob = prior + gaussian_log_prob(x_test, mean, var)  # 计算当前类别的后验对数得分，输出是标量。
    log_probs.append(log_prob)  # 把当前类别的得分加入列表。

pred = classes[np.argmax(log_probs)]  # 选取得分最大的类别作为预测结果。
print(\"Predicted class:\", int(pred))  # 打印最终预测类别。
