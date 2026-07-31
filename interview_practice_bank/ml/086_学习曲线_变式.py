"""
题目 086：学习曲线_变式

要求：完成“学习曲线”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建训练样本规模。
2. 训练分数随样本增加可能下降。
3. 计算泛化差距。
4. 判断增加数据是否改善高方差。
5. 设置期望泛化差距。
6. 根据目标判断是否继续收集数据。

完成标准：
- 验证学习曲线趋势。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
train_sizes = np.array([20, 50, 100, 200])  # 创建训练样本规模。
train_scores = np.array([0.99, 0.95, 0.91, 0.89])  # 训练分数随样本增加可能下降。
validation_scores = np.array([0.60, 0.72, 0.80, 0.85])  # 验证分数随样本增加改善。
gaps = train_scores - validation_scores  # 计算泛化差距。
more_data_helped = validation_scores[-1] > validation_scores[0] and gaps[-1] < gaps[0]  # 判断增加数据是否改善高方差。
target_gap = 0.06  # 设置期望泛化差距。
needs_more_data = gaps[-1] > target_gap  # 根据目标判断是否继续收集数据。
assert more_data_helped  # 验证学习曲线趋势。
print(train_sizes, gaps, needs_more_data)  # 输出学习曲线诊断。
