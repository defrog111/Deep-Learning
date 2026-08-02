"""
题目 018：逻辑回归与Sigmoid_变式

要求：完成“逻辑回归与Sigmoid”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 减去每行最大值稳定计算softmax。
3. 验证多分类概率。

完成标准：
- 验证多分类概率。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
multi_logits = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]); shifted_logits = multi_logits - multi_logits.max(axis=1, keepdims=True); softmax = np.exp(shifted_logits) / np.exp(shifted_logits).sum(axis=1, keepdims=True)  # 减去每行最大值稳定计算softmax。
assert np.allclose(softmax.sum(1), 1) and softmax.argmax(1).tolist() == [0, 2]  # 验证多分类概率。
