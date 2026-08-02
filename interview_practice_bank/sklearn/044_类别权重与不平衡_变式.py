"""
题目 044：类别权重与不平衡_变式

要求：完成“类别权重与不平衡”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 创建严重不平衡标签。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 类别权重可映射成每条样本权重供fit使用。

完成标准：
- 验证balanced权重规模。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.linear_model import LogisticRegression  # 导入分类器。
from sklearn.metrics import balanced_accuracy_score  # 导入平衡准确率。
labels = np.array([0] * 100 + [1] * 10)  # 创建严重不平衡标签。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight  # 导入类别和样本权重工具。
classes = np.unique(labels); class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels); sample_weights = compute_sample_weight(class_weight='balanced', y=labels)  # 类别权重可映射成每条样本权重供fit使用。
assert len(class_weights) == len(classes) and len(sample_weights) == len(labels) and np.isclose(sample_weights.mean(), 1)  # 验证balanced权重规模。
