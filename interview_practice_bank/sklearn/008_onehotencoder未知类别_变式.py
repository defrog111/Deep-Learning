"""
题目 008：OneHotEncoder未知类别_变式

要求：完成“OneHotEncoder未知类别”的综合题，说明fit、transform和predict各自只能使用哪些数据。

操作步骤：
1. 导入 NumPy。
2. 导入独热编码器。
3. 综合题统一导入NumPy用于shape、数值和标签检查。
4. 导入有序类别编码器。
5. 显式给定真实顺序并处理未知类别。
6. 验证有序编码和未知值。

完成标准：
- 验证有序编码和未知值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
from sklearn.preprocessing import OneHotEncoder  # 导入独热编码器。
import numpy as np  # 综合题统一导入NumPy用于shape、数值和标签检查。
from sklearn.preprocessing import OrdinalEncoder  # 导入有序类别编码器。
ordered_train = np.array([['low'], ['medium'], ['high']]); ordinal = OrdinalEncoder(categories=[['low', 'medium', 'high']], handle_unknown='use_encoded_value', unknown_value=-1).fit(ordered_train); ordinal_test = ordinal.transform([['medium'], ['unknown']])  # 显式给定真实顺序并处理未知类别。
assert ordinal_test.ravel().tolist() == [1.0, -1.0]  # 验证有序编码和未知值。
