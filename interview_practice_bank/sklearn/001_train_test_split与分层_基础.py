"""
题目 001：train_test_split与分层_基础

要求：完成“train_test_split与分层”的基础题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入经典分类数据。
2. 导入数据切分函数。
3. 加载特征和标签。
4. 分层切分保持类别比例。
5. 验证切分大小。
6. 输出shape和训练类别数。

完成标准：
- 验证切分大小。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from sklearn.datasets import load_iris  # 导入经典分类数据。
from sklearn.model_selection import train_test_split  # 导入数据切分函数。
features, labels = load_iris(return_X_y=True)  # 加载特征和标签。
train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=41, stratify=labels)  # 分层切分保持类别比例。
assert len(train_x) == 120 and len(test_x) == 30  # 验证切分大小。
print(train_x.shape, test_x.shape, __import__('numpy').bincount(train_y))  # 输出shape和训练类别数。
