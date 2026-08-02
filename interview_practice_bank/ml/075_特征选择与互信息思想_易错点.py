"""
题目 075：特征选择与互信息思想_易错点

要求：完成“特征选择与互信息思想”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. wrapper方法必须在验证或CV内部选特征数。
3. 防止按训练分数选择全部特征。

完成标准：
- 防止按训练分数选择全部特征。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
selection_train_scores = np.array([0.8, 0.9, 1.0]); selection_validation_scores = np.array([0.78, 0.85, 0.7]); selected_feature_count = np.array([1, 2, 3])[selection_validation_scores.argmax()]  # wrapper方法必须在验证或CV内部选特征数。
assert selected_feature_count == 2  # 防止按训练分数选择全部特征。
