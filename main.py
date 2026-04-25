# -*- coding: utf-8 -*-  # 声明编码，方便中文注释正常显示。
\"\"\"Extra interview problem: naive Conv2D forward.\"\"\"  # 说明这道题是手写二维卷积前向传播。

import numpy as np  # 导入 NumPy，用来做最朴素的循环卷积。

x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)  # 定义输入特征图，shape = (3, 3)。
kernel = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)  # 定义卷积核矩阵，shape = (2, 2)。
out_h = x.shape[0] - kernel.shape[0] + 1  # 计算输出高度 out_h = 2。
out_w = x.shape[1] - kernel.shape[1] + 1  # 计算输出宽度 out_w = 2。
out = np.zeros((out_h, out_w), dtype=np.float32)  # 创建输出矩阵，shape = (2, 2)。
for i in range(out_h):  # 逐行滑动卷积窗口。
    for j in range(out_w):  # 逐列滑动卷积窗口。
        region = x[i:i + kernel.shape[0], j:j + kernel.shape[1]]  # 取当前局部区域，shape = (2, 2)。
        out[i, j] = np.sum(region * kernel)  # 做逐元素乘法再求和，得到当前位置卷积结果。

print(\"Output shape:\", out.shape)  # 打印卷积输出的 shape。
print(\"Output:\n\", out)  # 打印卷积输出矩阵。
