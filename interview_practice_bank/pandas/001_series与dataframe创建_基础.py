"""
题目 001：Series与DataFrame创建_基础

要求：完成“Series与DataFrame创建”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 导入 Pandas。
2. 创建带标签和名称的整数Series。
3. 从对齐的Series和布尔结果创建DataFrame。
4. 验证Series与DataFrame的shape。
5. 验证标签索引被保留。
6. 输出基础对象和dtype。

完成标准：
- 验证Series与DataFrame的shape。
- 验证标签索引被保留。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
series = pd.Series([88, 92, 79], index=['Ada', 'Bob', 'Cyd'], name='score', dtype='int64')  # 创建带标签和名称的整数Series。
frame = pd.DataFrame({'score': series, 'passed': series.ge(80)})  # 从对齐的Series和布尔结果创建DataFrame。
assert series.shape == (3,) and frame.shape == (3, 2)  # 验证Series与DataFrame的shape。
assert frame.index.tolist() == ['Ada', 'Bob', 'Cyd']  # 验证标签索引被保留。
print(series, frame, frame.dtypes)  # 输出基础对象和dtype。
