"""
题目 016：列表字典集合推导式_综合

要求：完成“列表字典集合推导式”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 综合嵌套列表和字典推导式。
2. 验证推导结果。

完成标准：
- 验证推导结果。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

flattened = [item for row in [[1, 2], [3, 4]] for item in row]; indexed = {value: index for index, value in enumerate(flattened)}  # 综合嵌套列表和字典推导式。
assert flattened == [1, 2, 3, 4] and indexed[4] == 3  # 验证推导结果。
