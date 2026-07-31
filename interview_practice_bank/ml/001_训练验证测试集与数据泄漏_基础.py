"""
题目 001：训练验证测试集与数据泄漏_基础

要求：完成“训练验证测试集与数据泄漏”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 固定切分随机性。
2. 先打乱样本索引。
3. 按60/20/20切分。
4. 构造单特征数据。
5. 只在训练集拟合预处理统计量。
6. 用训练统计量转换测试集。

完成标准：
- 验证集合互斥和测试样本数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
rng = np.random.default_rng(42)  # 固定切分随机性。
indices = rng.permutation(100)  # 先打乱样本索引。
train, valid, test = indices[:60], indices[60:80], indices[80:]  # 按60/20/20切分。
features = np.arange(100, dtype=float)  # 构造单特征数据。
train_mean = features[train].mean()  # 只在训练集拟合预处理统计量。
standardized_test = features[test] - train_mean  # 用训练统计量转换测试集。
assert not set(train) & set(test) and len(standardized_test) == 20  # 验证集合互斥和测试样本数。
print(train_mean, standardized_test[:3])  # 输出无泄漏处理结果。
