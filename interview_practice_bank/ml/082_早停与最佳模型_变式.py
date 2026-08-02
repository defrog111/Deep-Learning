"""
题目 082：早停与最佳模型_变式

要求：完成“早停与最佳模型”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. min_delta过滤微小噪声改进。
for epoch, value in enumerate(validation_history):  # 逐轮监控。
    if value < best_value - min_delta: best_value, patience_count = value, 0  # 显著改进时保存。
    else: patience_count += 1  # 否则累计无改进轮数。
    if patience_count >= 2: stop_epoch = epoch; break  # 达到patience时停止。
3. 验证min_delta和patience共同作用。

完成标准：
- 验证min_delta和patience共同作用。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
validation_history = np.array([1.0, 0.8, 0.795, 0.792, 0.791]); min_delta = 0.01; patience_count = 0; best_value = np.inf; stop_epoch = None  # min_delta过滤微小噪声改进。
for epoch, value in enumerate(validation_history):  # 逐轮监控。
    if value < best_value - min_delta: best_value, patience_count = value, 0  # 显著改进时保存。
    else: patience_count += 1  # 否则累计无改进轮数。
    if patience_count >= 2: stop_epoch = epoch; break  # 达到patience时停止。
assert stop_epoch == 3  # 验证min_delta和patience共同作用。
