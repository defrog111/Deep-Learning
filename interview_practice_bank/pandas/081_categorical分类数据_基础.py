"""
题目 081：Categorical分类数据_基础

要求：完成“Categorical分类数据”的基础题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 定义有序分类类型。
2. 构造任务表。
3. 转为有序Categorical。
4. 按业务顺序而非字母顺序排序。
5. 获取内部整数编码。

完成标准：
- 验证排序顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
priority = pd.CategoricalDtype(['low', 'medium', 'high'], ordered=True)  # 定义有序分类类型。
frame = pd.DataFrame({'task': list('ABCD'), 'priority': ['high', 'low', 'medium', 'low']})  # 构造任务表。
frame['priority'] = frame['priority'].astype(priority)  # 转为有序Categorical。
sorted_frame = frame.sort_values('priority')  # 按业务顺序而非字母顺序排序。
codes = frame['priority'].cat.codes  # 获取内部整数编码。
assert sorted_frame.iloc[0]['priority'] == 'low'  # 验证排序顺序。
print(sorted_frame, '\n', codes)  # 输出分类排序和编码。
