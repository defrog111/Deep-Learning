"""
题目 087：explode展开列表列_易错点

要求：完成“explode展开列表列”的易错点题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造列表列。
2. 同时展开等长列表列。
3. explode后恢复数值类型。

完成标准：
- 验证一行多值已展开。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
frame = pd.DataFrame({'order': [1, 2], 'items': [['pen', 'book'], ['desk']], 'quantities': [[2, 1], [1]]})  # 构造列表列。
long = frame.explode(['items', 'quantities'], ignore_index=True)  # 同时展开等长列表列。
long['quantities'] = long['quantities'].astype(int)  # explode后恢复数值类型。
assert len(long) == 3  # 验证一行多值已展开。
print(long)  # 输出规范化长表。
