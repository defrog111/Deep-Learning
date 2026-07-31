"""
题目 090：FFT频域分析_变式

要求：完成“FFT频域分析”的变式题，写出关键数组的shape并解释结果。

操作步骤：
1. 设置采样点数。
2. 构造一秒内的均匀时间点。
3. 设置正弦信号频率。
4. 生成单频正弦信号。
5. 对实信号计算单边FFT。
6. 计算每个频点对应频率。
7. 找到幅度最大的主频。

完成标准：
- 验证FFT识别出真实频率。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
sample_count = 128  # 设置采样点数。
time = np.arange(sample_count) / sample_count  # 构造一秒内的均匀时间点。
frequency = 4  # 设置正弦信号频率。
signal = np.sin(2 * np.pi * frequency * time)  # 生成单频正弦信号。
spectrum = np.fft.rfft(signal)  # 对实信号计算单边FFT。
frequencies = np.fft.rfftfreq(sample_count, d=1 / sample_count)  # 计算每个频点对应频率。
dominant = frequencies[np.abs(spectrum).argmax()]  # 找到幅度最大的主频。
assert dominant == frequency  # 验证FFT识别出真实频率。
print(dominant)  # 输出主频。
