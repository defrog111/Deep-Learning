"""
题目 077：概率校准与Brier分数_基础

要求：完成“概率校准与Brier分数”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建模型概率。
2. 创建真实标签。
3. Brier分数衡量概率校准和准确性。
4. 转为硬分类仅用于accuracy。
5. 计算准确率。
6. 构造更极端的概率。
7. 比较校准误差。

完成标准：
- 验证Brier范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
probabilities = np.array([0.1, 0.4, 0.8, 0.9])  # 创建模型概率。
targets = np.array([0.0, 1.0, 1.0, 1.0])  # 创建真实标签。
brier = np.mean((probabilities - targets) ** 2)  # Brier分数衡量概率校准和准确性。
hard_predictions = probabilities >= 0.5  # 转为硬分类仅用于accuracy。
accuracy = np.mean(hard_predictions == targets)  # 计算准确率。
overconfident = np.clip(probabilities * 1.1, 0, 1)  # 构造更极端的概率。
overconfident_brier = np.mean((overconfident - targets) ** 2)  # 比较校准误差。
assert 0 <= brier <= 1  # 验证Brier范围。
print(brier, accuracy, overconfident_brier)  # 输出概率与硬指标。
