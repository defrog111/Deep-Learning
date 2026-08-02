"""
题目 046：学习曲线_变式

要求：完成“学习曲线”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 加载数据。
2. 综合题统一导入NumPy用于shape、数值和标签检查。
3. 与learning_curve不同，validation_curve横轴是超参数。

完成标准：
- 验证参数候选数乘折数。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.datasets import load_iris  # 导入分类数据。
from sklearn.tree import DecisionTreeClassifier  # 导入对小样本规模稳定的模型。
from sklearn.model_selection import learning_curve  # 导入学习曲线工具。
features, labels = load_iris(return_X_y=True)  # 加载数据。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.model_selection import validation_curve  # 导入单超参数诊断曲线。
from sklearn.tree import DecisionTreeClassifier  # 导入可调深度模型。
train_curve_scores, valid_curve_scores = validation_curve(DecisionTreeClassifier(random_state=42), features, labels, param_name='max_depth', param_range=[1, 2, 4, None], cv=3, scoring='accuracy')  # 与learning_curve不同，validation_curve横轴是超参数。
assert train_curve_scores.shape == valid_curve_scores.shape == (4, 3)  # 验证参数候选数乘折数。
