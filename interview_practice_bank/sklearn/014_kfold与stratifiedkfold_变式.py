"""
题目 014：KFold与StratifiedKFold_变式

要求：完成“KFold与StratifiedKFold”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入 NumPy。
2. 导入两种K折。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 导入组K折和时间序列切分。
5. 根据数据依赖结构选择CV。
6. 验证组隔离与时间gap。

完成标准：
- 验证组隔离与时间gap。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.model_selection import KFold, StratifiedKFold  # 导入两种K折。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.model_selection import GroupKFold, TimeSeriesSplit  # 导入组K折和时间序列切分。
group_features = np.zeros((12, 1)); group_labels = np.array([0, 1] * 6); group_ids = np.repeat(np.arange(6), 2); grouped_folds = list(GroupKFold(n_splits=3).split(group_features, group_labels, group_ids)); temporal_folds = list(TimeSeriesSplit(n_splits=3, gap=1).split(group_features))  # 根据数据依赖结构选择CV。
assert all(not set(group_ids[train]) & set(group_ids[test]) for train, test in grouped_folds) and all(train.max() + 1 < test.min() for train, test in temporal_folds)  # 验证组隔离与时间gap。
