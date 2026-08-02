"""
题目 091：FFT频域分析_易错点

要求：完成“FFT频域分析”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 导入 NumPy。
2. fftshift把零频移动到中心。
3. 验证双边频谱shape。

完成标准：
- 验证双边频谱shape。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
shifted_frequency = np.fft.fftshift(np.fft.fftfreq(8)); shifted_spectrum = np.fft.fftshift(np.fft.fft(np.arange(8)))  # fftshift把零频移动到中心。
assert shifted_frequency.shape == shifted_spectrum.shape == (8,)  # 验证双边频谱shape。
