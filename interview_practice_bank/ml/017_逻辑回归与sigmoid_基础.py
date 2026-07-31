"""
题目 017：逻辑回归与Sigmoid_基础

要求：完成“逻辑回归与Sigmoid”的基础题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造模型logits。
2. 使用Sigmoid转换为正类概率。
3. 定义二分类标签。
4. 防止对0取log。
5. 手算二元交叉熵。
6. 使用阈值转类别。

完成标准：
- 验证概率范围。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
logits = np.array([-3.0, 0.0, 2.0])  # 构造模型logits。
probabilities = 1 / (1 + np.exp(-logits))  # 使用Sigmoid转换为正类概率。
labels = np.array([0.0, 1.0, 1.0])  # 定义二分类标签。
epsilon = 1e-12  # 防止对0取log。
log_loss = -(labels * np.log(probabilities + epsilon) + (1 - labels) * np.log(1 - probabilities + epsilon)).mean()  # 手算二元交叉熵。
predictions = probabilities >= 0.45  # 使用阈值转类别。
assert np.all((0 < probabilities) & (probabilities < 1))  # 验证概率范围。
print(probabilities, predictions, log_loss)  # 输出逻辑回归结果。
