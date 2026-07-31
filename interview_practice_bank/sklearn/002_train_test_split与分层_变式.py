"""
题目 002：train_test_split与分层_变式

要求：完成“train_test_split与分层”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载特征和标签。
2. 分层切分保持类别比例。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 同一病人或用户的多条记录不能跨集合。

完成标准：
- 验证切分大小。
- 验证组级隔离。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入经典分类数据。
from sklearn.model_selection import train_test_split  # 导入数据切分函数。
features, labels = load_iris(return_X_y=True)  # 加载特征和标签。
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)  # 分层切分保持类别比例。
assert len(train_x) == 120 and len(test_x) == 30  # 验证切分大小。
print(train_x.shape, test_x.shape, __import__('numpy').bincount(train_y))  # 输出shape和训练类别数。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.model_selection import GroupShuffleSplit  # 导入按组随机切分工具。
groups = np.repeat(np.arange(50), 3); group_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42); grouped_train, grouped_test = next(group_splitter.split(features, labels, groups))  # 同一病人或用户的多条记录不能跨集合。
assert not set(groups[grouped_train]) & set(groups[grouped_test])  # 验证组级隔离。
