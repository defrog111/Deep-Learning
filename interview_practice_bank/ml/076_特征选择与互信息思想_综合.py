"""
题目 076：特征选择与互信息思想_综合

要求：完成“特征选择与互信息思想”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 排列重要性衡量打乱单列后的性能下降。
3. 综合得到模型无关重要性排序。

完成标准：
- 综合得到模型无关重要性排序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
baseline_metric = 0.9; shuffled_metrics = np.array([0.89, 0.6, 0.88]); permutation_importance = baseline_metric - shuffled_metrics; ranked_features = np.argsort(permutation_importance)[::-1]  # 排列重要性衡量打乱单列后的性能下降。
assert ranked_features[0] == 1 and permutation_importance[1] == permutation_importance.max()  # 综合得到模型无关重要性排序。
