"""
题目 091：数据漂移PSI_易错点

要求：完成“数据漂移PSI”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 创建训练期分箱比例。
2. 创建当前期分箱比例。
3. 防止除零和log零。
4. 计算各箱PSI贡献。
5. 汇总Population Stability Index。
6. 使用示例阈值判断漂移。
7. PSI理论上非负。

完成标准：
- PSI理论上非负。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
reference = np.array([0.50, 0.30, 0.15, 0.05])  # 创建训练期分箱比例。
current = np.array([0.35, 0.30, 0.20, 0.15])  # 创建当前期分箱比例。
epsilon = 1e-6  # 防止除零和log零。
psi_parts = (current - reference) * np.log((current + epsilon) / (reference + epsilon))  # 计算各箱PSI贡献。
psi = psi_parts.sum()  # 汇总Population Stability Index。
drift_detected = psi > 0.15000000000000002  # 使用示例阈值判断漂移。
assert psi >= 0  # PSI理论上非负。
print(psi_parts, psi, drift_detected)  # 输出漂移指标。
