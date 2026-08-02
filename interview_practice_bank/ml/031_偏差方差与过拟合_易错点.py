"""
题目 031：偏差方差与过拟合_易错点

要求：完成“偏差方差与过拟合”的易错点题，计算结果并回答为什么不能使用错误做法。

操作步骤：
1. 导入 NumPy。
2. 若数组代表测试误差，用它选复杂度会造成测试集泄漏。
3. 识别高方差与错误选择动作。

完成标准：
- 识别高方差与错误选择动作。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
training_errors = np.array([0.5, 0.2, 0.05]); validation_errors = np.array([0.55, 0.3, 0.8]); gaps = validation_errors - training_errors; selected_by_test_wrong = validation_errors.argmin()  # 若数组代表测试误差，用它选复杂度会造成测试集泄漏。
assert gaps[-1] > gaps[0] and selected_by_test_wrong == 1  # 识别高方差与错误选择动作。
