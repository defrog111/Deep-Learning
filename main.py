# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Core interview problem: one-split decision tree with Gini.\"\"\"  # 说明这道题是用 Gini 找最优分裂。

import numpy as np  # 导入 NumPy，用来做切片和类别统计。

def gini(y):  # 定义 Gini impurity 函数，输入 y 的 shape = (num_samples,)。
    _, counts = np.unique(y, return_counts=True)  # 统计每个类别出现次数，counts 的 shape = (num_classes,)。
    probs = counts / np.sum(counts)  # 计算类别概率向量，shape = (num_classes,)。
    return 1.0 - np.sum(probs ** 2)  # 返回 Gini impurity，输出是标量。

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)  # 定义一维特征向量，shape = (5,)。
y = np.array([0, 0, 1, 1, 1], dtype=np.int32)  # 定义类别标签向量，shape = (5,)。
thresholds = [1.5, 2.5, 3.5, 4.5]  # 定义候选分裂阈值列表。
best_threshold = None  # 初始化最优阈值占位变量。
best_score = float('inf')  # 初始化最优分裂得分为正无穷。

for threshold in thresholds:  # 遍历每个候选阈值。
    left_y = y[x <= threshold]  # 取左子树标签向量，shape = (num_left_samples,)。
    right_y = y[x > threshold]  # 取右子树标签向量，shape = (num_right_samples,)。
    score = (len(left_y) / len(y)) * gini(left_y) + (len(right_y) / len(y)) * gini(right_y)  # 计算加权 Gini，输出是标量。
    if score < best_score:  # 如果当前分裂更好，就更新最优答案。
        best_score = score  # 保存更好的 Gini 分数。
        best_threshold = threshold  # 保存对应阈值。

print(\"Best threshold:\", best_threshold)  # 打印最优分裂阈值。
print(\"Best gini score:\", best_score)  # 打印最优分裂分数。
