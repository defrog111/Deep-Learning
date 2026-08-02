"""
题目 092：FFT频域分析_综合

要求：完成“FFT频域分析”的综合题，写出关键数组的shape并解释结果。

操作步骤：
1. 设置采样点数。
2. 构造一秒内的均匀时间点。
3. 设置正弦信号频率。
4. 生成单频正弦信号。
5. 对实信号计算单边FFT。
6. 综合使用窗函数减轻非整周期频谱泄漏。

完成标准：
- 验证加窗不改变频点数量。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
sample_count = 128  # 设置采样点数。
time = np.arange(sample_count) / sample_count  # 构造一秒内的均匀时间点。
frequency = 6  # 设置正弦信号频率。
signal = np.sin(2 * np.pi * frequency * time)  # 生成单频正弦信号。
spectrum = np.fft.rfft(signal)  # 对实信号计算单边FFT。
window = np.hanning(sample_count); windowed_spectrum = np.fft.rfft(signal * window)  # 综合使用窗函数减轻非整周期频谱泄漏。
assert windowed_spectrum.shape == spectrum.shape  # 验证加窗不改变频点数量。
