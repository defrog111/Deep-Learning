"""
题目 055：决策树Gini与Entropy_易错点

要求：完成“决策树Gini与Entropy”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 不能用训练准确率选择树深，否则总偏向复杂树。
3. 验证预剪枝需看验证集。

完成标准：
- 验证预剪枝需看验证集。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
deep_train_scores = np.array([0.8, 0.95, 1.0]); deep_valid_scores = np.array([0.78, 0.82, 0.65]); best_depth_index = deep_valid_scores.argmax()  # 不能用训练准确率选择树深，否则总偏向复杂树。
assert best_depth_index == 1 and deep_train_scores.argmax() != best_depth_index  # 验证预剪枝需看验证集。
