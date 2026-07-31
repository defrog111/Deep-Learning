"""
题目 090：FFT频域分析_变式

要求：完成“FFT频域分析”的变式题，写出关键数组的shape并解释结果。
先自己实现，再运行本文件查看参考代码结果。
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
