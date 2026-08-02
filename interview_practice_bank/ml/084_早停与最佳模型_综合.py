"""
题目 084：早停与最佳模型_综合

要求：完成“早停与最佳模型”的综合题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 综合保存并恢复最佳epoch而非最后epoch。
3. 验证最佳模型恢复。

完成标准：
- 验证最佳模型恢复。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
model_states = [{'weight': value} for value in [0.1, 0.2, 0.3, 0.4]]; validation_losses = np.array([0.9, 0.6, 0.5, 0.8]); best_epoch = validation_losses.argmin(); restored_state = dict(model_states[best_epoch])  # 综合保存并恢复最佳epoch而非最后epoch。
assert restored_state == {'weight': 0.3} and best_epoch != len(model_states) - 1  # 验证最佳模型恢复。
