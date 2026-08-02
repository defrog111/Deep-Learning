"""
题目 020：sorted的key与稳定排序_综合

要求：完成“sorted的key与稳定排序”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入旧式比较函数适配器。
2. 把cmp转换为key。
3. 验证自定义比较。

完成标准：
- 验证自定义比较。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import cmp_to_key  # 导入旧式比较函数适配器。
by_last_digit = sorted([21, 13, 32], key=cmp_to_key(lambda left, right: left % 10 - right % 10))  # 把cmp转换为key。
assert by_last_digit == [21, 32, 13]  # 验证自定义比较。
