"""
题目 061：melt与宽长表转换_基础

要求：完成“melt与宽长表转换”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 构造宽表。
2. 宽表转长表。
3. 长表恢复宽表。
4. 清除列索引名称便于比较。

完成标准：
- 验证往返转换保持数据。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
wide = pd.DataFrame({'id': [1, 2], 'math': [90, 80], 'english': [85, 88]})  # 构造宽表。
long = wide.melt(id_vars='id', var_name='subject', value_name='score')  # 宽表转长表。
restored = long.pivot(index='id', columns='subject', values='score').reset_index()  # 长表恢复宽表。
restored.columns.name = None  # 清除列索引名称便于比较。
assert restored[['id', 'math', 'english']].equals(wide)  # 验证往返转换保持数据。
print(long, '\n', restored)  # 输出两种形态。
