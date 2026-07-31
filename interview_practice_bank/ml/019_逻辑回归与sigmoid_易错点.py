"""
题目 019：逻辑回归与Sigmoid_易错点

要求：完成“逻辑回归与Sigmoid”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造模型logits。
2. 使用Sigmoid转换为正类概率。
3. 定义二分类标签。
4. 防止对0取log。
5. 手算二元交叉熵。
6. 使用阈值转类别。
7. 直接按logits计算BCE避免sigmoid上溢下溢。

完成标准：
- 验证概率范围。
- 验证数值稳定性。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
logits = np.array([-3.0, 0.0, 2.0])  # 构造模型logits。
probabilities = 1 / (1 + np.exp(-logits))  # 使用Sigmoid转换为正类概率。
labels = np.array([0.0, 1.0, 1.0])  # 定义二分类标签。
epsilon = 1e-12  # 防止对0取log。
log_loss = -(labels * np.log(probabilities + epsilon) + (1 - labels) * np.log(1 - probabilities + epsilon)).mean()  # 手算二元交叉熵。
predictions = probabilities >= 0.55  # 使用阈值转类别。
assert np.all((0 < probabilities) & (probabilities < 1))  # 验证概率范围。
print(probabilities, predictions, log_loss)  # 输出逻辑回归结果。
extreme_logits = np.array([-1000.0, 1000.0]); stable_losses = np.maximum(extreme_logits, 0) - extreme_logits * np.array([0.0, 1.0]) + np.log1p(np.exp(-np.abs(extreme_logits)))  # 直接按logits计算BCE避免sigmoid上溢下溢。
assert np.isfinite(stable_losses).all() and np.allclose(stable_losses, 0)  # 验证数值稳定性。
