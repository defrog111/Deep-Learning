"""
题目 043：where与select条件选择_易错点

要求：完成“where与select条件选择”的易错点题，写出关键数组的shape并解释结果。

操作步骤：
1. 构造跨多个条件区间的数据。
2. 二选一条件向量化。
3. 多条件按顺序选择。
4. 把数值限制到闭区间。
5. 临时管理可预期的浮点告警。
    eager = np.where(values != 0, 1 / values, 0)  # where会先计算两个分支，不能阻止除零计算。
6. 使用errstate管理已知浮点告警。

完成标准：
- 验证条件优先级。
- 使用errstate管理已知浮点告警。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import numpy as np  # 导入 NumPy。
values = np.array([-3, 0, 2, 8, 12])  # 构造跨多个条件区间的数据。
sign = np.where(values >= 0, 1, -1)  # 二选一条件向量化。
labels = np.select([values < 0, values < 10], ['negative', 'small'], default='large')  # 多条件按顺序选择。
clipped = np.clip(values, 0, 10)  # 把数值限制到闭区间。
assert labels.tolist() == ['negative', 'small', 'small', 'small', 'large']  # 验证条件优先级。
print(sign, labels, clipped)  # 输出条件运算结果。
with np.errstate(divide='ignore', invalid='ignore'):  # 临时管理可预期的浮点告警。
    eager = np.where(values != 0, 1 / values, 0)  # where会先计算两个分支，不能阻止除零计算。
assert np.isfinite(eager).all()  # 使用errstate管理已知浮点告警。
