"""
题目 060：Bagging与Boosting_综合

要求：完成“Bagging与Boosting”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. Stacking元学习器必须训练在out-of-fold预测上。
3. 综合验证两层模型输入。

完成标准：
- 综合验证两层模型输入。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
base_oof = np.array([[0.8, 0.6], [0.2, 0.3], [0.7, 0.9], [0.1, 0.4]]); meta_weights = np.linalg.lstsq(np.c_[np.ones(4), base_oof], np.array([1, 0, 1, 0]), rcond=None)[0]; stacked = np.c_[np.ones(4), base_oof] @ meta_weights  # Stacking元学习器必须训练在out-of-fold预测上。
assert stacked.shape == (4,) and np.isfinite(stacked).all()  # 综合验证两层模型输入。
