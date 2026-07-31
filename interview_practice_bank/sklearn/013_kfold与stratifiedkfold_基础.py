"""
题目 013：KFold与StratifiedKFold_基础

要求：完成“KFold与StratifiedKFold”的基础题，说明fit、transform和predict各自只能使用哪些数据。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
from sklearn.model_selection import KFold, StratifiedKFold  # 导入两种K折。
features = np.zeros((12, 1))  # 创建占位特征。
labels = np.array([0] * 8 + [1] * 4)  # 创建不平衡标签。
plain = KFold(n_splits=4, shuffle=True, random_state=42)  # 普通KFold不保证每折类别比例。
stratified = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  # 分层KFold保持比例。
ratios = [labels[test].mean() for _, test in stratified.split(features, labels)]  # 计算各验证折正类比例。
assert np.allclose(ratios, labels.mean())  # 验证分层比例一致。
print(len(list(plain.split(features))), ratios)  # 输出折数和分层比例。
