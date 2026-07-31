"""
题目 035：交叉验证_易错点

要求：完成“交叉验证”的易错点题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
indices = np.arange(12)  # 创建样本索引。
fold_count = 5  # 设置K折数量。
folds = np.array_split(indices, fold_count)  # 把样本近似等分。
validation_seen = []  # 记录各折验证样本。
for fold in range(fold_count):  # 每一折轮流作为验证集。
    validation = folds[fold]  # 选择当前验证折。
    training = np.concatenate([part for index, part in enumerate(folds) if index != fold])  # 合并其余训练折。
    assert not set(training) & set(validation)  # 验证单折无交集。
    validation_seen.extend(validation.tolist())  # 收集验证覆盖。
assert sorted(validation_seen) == indices.tolist()  # 验证每个样本恰做一次验证。
print([len(fold) for fold in folds])  # 输出各折大小。
