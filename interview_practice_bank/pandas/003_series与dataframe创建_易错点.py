"""
题目 003：Series与DataFrame创建_易错点

要求：完成“Series与DataFrame创建”的易错点题，并解释输出的 shape、索引和数据类型。

操作步骤：
1. 创建价格Series。
2. 创建索引集合不同的数量Series。
3. Pandas按标签对齐，只有共同标签B得到非缺失结果。
4. fill_value显式处理仅出现在一侧的标签。
5. 转NumPy后按位置运算但会丢失标签语义。
6. 对比标签对齐与位置计算。

完成标准：
- 验证自动索引并集和NaN陷阱。
- 验证带填充值的标签运算。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import pandas as pd  # 导入 Pandas。
prices = pd.Series([10, 20], index=['A', 'B'], dtype='int64')  # 创建价格Series。
quantities = pd.Series([2, 3], index=['B', 'C'], dtype='int64')  # 创建索引集合不同的数量Series。
naive_total = prices * quantities  # Pandas按标签对齐，只有共同标签B得到非缺失结果。
safe_total = prices.mul(quantities, fill_value=0)  # fill_value显式处理仅出现在一侧的标签。
positional_total = prices.to_numpy() * quantities.to_numpy()  # 转NumPy后按位置运算但会丢失标签语义。
assert naive_total.index.tolist() == ['A', 'B', 'C'] and naive_total.notna().sum() == 1  # 验证自动索引并集和NaN陷阱。
assert safe_total.to_dict() == {'A': 0.0, 'B': 40.0, 'C': 0.0}  # 验证带填充值的标签运算。
print(naive_total, safe_total, positional_total)  # 对比标签对齐与位置计算。
