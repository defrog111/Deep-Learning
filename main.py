# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: NumPy dropout.\"\"\"  # 说明这道题是手写 Dropout。

import numpy as np  # 导入 NumPy，用来生成掩码并做逐元素乘法。

x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 定义输入矩阵，shape = (2, 2)。
drop_prob = 0.5  # 定义 Dropout 概率，表示每个位置有 50% 概率被丢弃。
keep_prob = 1.0 - drop_prob  # 定义保留概率，是标量。
rng = np.random.default_rng(seed=42)  # 创建随机数生成器，方便结果可复现。
mask = (rng.random(x.shape) < keep_prob).astype(np.float32)  # 生成 Dropout 掩码矩阵，shape = (2, 2)。
out_train = x * mask / keep_prob  # 在训练阶段做随机失活并做反向缩放，shape = (2, 2)。
out_test = x  # 在测试阶段直接使用原输入，shape = (2, 2)。

print(\"Mask shape:\", mask.shape)  # 打印掩码矩阵的 shape。
print(\"Train output shape:\", out_train.shape)  # 打印训练阶段输出的 shape。
print(\"Test output shape:\", out_test.shape)  # 打印测试阶段输出的 shape。
