"""
题目 003：Series与DataFrame创建_易错点

要求：完成“Series与DataFrame创建”的易错点题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建带标签的一维 Series。
2. 转为 DataFrame 并把索引恢复成普通列。
3. 使用向量化比较创建布尔列。

完成标准：
- 验证行列数符合预期。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
series = pd.Series([5, 6, 7], index=['a', 'b', 'c'], name='score')  # 创建带标签的一维 Series。
frame = series.to_frame().reset_index(names='student')  # 转为 DataFrame 并把索引恢复成普通列。
frame['passed'] = frame['score'].ge(4)  # 使用向量化比较创建布尔列。
assert frame.shape == (3, 3)  # 验证行列数符合预期。
print(frame)  # 输出结果用于检查。
