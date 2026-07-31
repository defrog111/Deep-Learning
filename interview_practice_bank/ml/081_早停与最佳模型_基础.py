"""
题目 081：早停与最佳模型_基础

要求：完成“早停与最佳模型”的基础题，计算结果并回答为什么不能使用错误做法。
先自己实现，再运行本文件查看参考代码结果。
"""

import numpy as np  # 导入 NumPy。
validation_loss = np.array([0.9, 0.7, 0.55, 0.50, 0.52, 0.58, 0.61])  # 构造验证损失轨迹。
patience = 1  # 设置连续无改进容忍轮数。
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
