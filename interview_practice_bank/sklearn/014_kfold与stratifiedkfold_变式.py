"""
题目 014：KFold与StratifiedKFold_变式

要求：完成“KFold与StratifiedKFold”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 创建占位特征。
2. 创建不平衡标签。
3. 普通KFold不保证每折类别比例。
4. 分层KFold保持比例。
5. 计算各验证折正类比例。

完成标准：
- 验证分层比例一致。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
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
