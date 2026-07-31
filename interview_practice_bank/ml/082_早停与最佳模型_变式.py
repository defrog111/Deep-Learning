"""
题目 082：早停与最佳模型_变式

要求：完成“早停与最佳模型”的变式题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 构造验证损失轨迹。
2. 设置连续无改进容忍轮数。
3. 初始化最佳损失。
4. 初始化最佳轮次。
5. 初始化等待计数。
6. 逐轮检查验证损失。
7. 发现严格改进。
8. 保存最佳状态并清零等待。
9. 当前轮没有改进。
10. 累加等待轮数。

完成标准：
- 验证最优模型出现在第3轮零基索引。
- 验证min_delta和patience共同作用。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
validation_loss = np.array([0.9, 0.7, 0.55, 0.50, 0.52, 0.58, 0.61])  # 构造验证损失轨迹。
patience = 2  # 设置连续无改进容忍轮数。
best_loss = np.inf  # 初始化最佳损失。
best_epoch = -1  # 初始化最佳轮次。
wait = 0  # 初始化等待计数。
for epoch, loss in enumerate(validation_loss):  # 逐轮检查验证损失。
    if loss < best_loss:  # 发现严格改进。
        best_loss, best_epoch, wait = loss, epoch, 0  # 保存最佳状态并清零等待。
    else:  # 当前轮没有改进。
        wait += 1  # 累加等待轮数。
        if wait >= patience:  # 达到patience则停止。
            break  # 退出训练循环。
assert best_epoch == 3  # 验证最优模型出现在第3轮零基索引。
print(best_epoch, best_loss, epoch)  # 输出最佳和停止轮次。
validation_history = np.array([1.0, 0.8, 0.795, 0.792, 0.791]); min_delta = 0.01; patience_count = 0; best_value = np.inf; stop_epoch = None  # min_delta过滤微小噪声改进。
for epoch, value in enumerate(validation_history):  # 逐轮监控。
    if value < best_value - min_delta: best_value, patience_count = value, 0  # 显著改进时保存。
    else: patience_count += 1  # 否则累计无改进轮数。
    if patience_count >= 2: stop_epoch = epoch; break  # 达到patience时停止。
assert stop_epoch == 3  # 验证min_delta和patience共同作用。
