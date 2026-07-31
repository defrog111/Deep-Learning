"""
题目 054：决策树Gini与Entropy_变式

要求：完成“决策树Gini与Entropy”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 表示节点中的两类样本数。
2. 计算类别概率。
3. 计算Gini不纯度。
4. 计算信息熵。
5. 构造左子节点类别计数。
6. 构造右子节点类别计数。
7. 计算切分后加权Gini。
8. 不纯度下降即切分收益。
9. 计算entropy信息增益。

完成标准：
- 验证该切分改善纯度。
- 验证有效切分不增加加权不纯度。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
counts = np.array([6.0, 4.0])  # 表示节点中的两类样本数。
probabilities = counts / counts.sum()  # 计算类别概率。
gini = 1 - np.sum(probabilities**2)  # 计算Gini不纯度。
entropy = -np.sum(probabilities * np.log2(probabilities, where=probabilities > 0))  # 计算信息熵。
left_counts = np.array([5.0, 1.0])  # 构造左子节点类别计数。
right_counts = counts - left_counts  # 构造右子节点类别计数。
weighted_gini = (left_counts.sum() * (1 - np.sum((left_counts / left_counts.sum()) ** 2)) + right_counts.sum() * (1 - np.sum((right_counts / right_counts.sum()) ** 2))) / counts.sum()  # 计算切分后加权Gini。
gain = gini - weighted_gini  # 不纯度下降即切分收益。
assert gain > 0  # 验证该切分改善纯度。
print(gini, entropy, weighted_gini, gain)  # 输出树切分指标。
parent_counts = np.array([6, 4]); left_counts = np.array([4, 1]); right_counts = parent_counts - left_counts; entropy_fn = lambda counts: -(counts[counts > 0] / counts.sum() * np.log2(counts[counts > 0] / counts.sum())).sum(); information_gain = entropy_fn(parent_counts) - (left_counts.sum() * entropy_fn(left_counts) + right_counts.sum() * entropy_fn(right_counts)) / parent_counts.sum()  # 计算entropy信息增益。
assert information_gain >= 0  # 验证有效切分不增加加权不纯度。
