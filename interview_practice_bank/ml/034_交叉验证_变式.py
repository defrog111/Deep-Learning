"""
题目 034：交叉验证_变式

要求：完成“交叉验证”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 手工构造保持类别比例的分层K折。
3. 验证每折比例。

完成标准：
- 验证每折比例。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
labels_cv = np.array([0] * 8 + [1] * 4); class_zero_cv = np.flatnonzero(labels_cv == 0); class_one_cv = np.flatnonzero(labels_cv == 1); stratified_folds = [np.r_[class_zero_cv[index::4], class_one_cv[index::4]] for index in range(4)]  # 手工构造保持类别比例的分层K折。
assert all(np.isclose(labels_cv[fold].mean(), labels_cv.mean()) for fold in stratified_folds)  # 验证每折比例。
