"""
题目 034：交叉验证_变式

要求：完成“交叉验证”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建样本索引。
2. 设置K折数量。
3. 把样本近似等分。
4. 记录各折验证样本。
5. 每一折轮流作为验证集。
6. 选择当前验证折。
7. 合并其余训练折。
8. 收集验证覆盖。
9. 手工构造保持类别比例的分层K折。

完成标准：
- 验证单折无交集。
- 验证每个样本恰做一次验证。
- 验证每折比例。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
indices = np.arange(12)  # 创建样本索引。
fold_count = 4  # 设置K折数量。
folds = np.array_split(indices, fold_count)  # 把样本近似等分。
validation_seen = []  # 记录各折验证样本。
for fold in range(fold_count):  # 每一折轮流作为验证集。
    validation = folds[fold]  # 选择当前验证折。
    training = np.concatenate([part for index, part in enumerate(folds) if index != fold])  # 合并其余训练折。
    assert not set(training) & set(validation)  # 验证单折无交集。
    validation_seen.extend(validation.tolist())  # 收集验证覆盖。
assert sorted(validation_seen) == indices.tolist()  # 验证每个样本恰做一次验证。
print([len(fold) for fold in folds])  # 输出各折大小。
labels_cv = np.array([0] * 8 + [1] * 4); class_zero_cv = np.flatnonzero(labels_cv == 0); class_one_cv = np.flatnonzero(labels_cv == 1); stratified_folds = [np.r_[class_zero_cv[index::4], class_one_cv[index::4]] for index in range(4)]  # 手工构造保持类别比例的分层K折。
assert all(np.isclose(labels_cv[fold].mean(), labels_cv.mean()) for fold in stratified_folds)  # 验证每折比例。
