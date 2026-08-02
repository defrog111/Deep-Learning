"""
题目 010：字典集合与哈希_变式

要求：完成“字典集合与哈希”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入带默认工厂的字典。
2. 按奇偶分组且无需手动判断键。
3. 验证defaultdict分组。

完成标准：
- 验证defaultdict分组。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import defaultdict  # 导入带默认工厂的字典。
grouped = defaultdict(list); [grouped[value % 2].append(value) for value in range(6)]  # 按奇偶分组且无需手动判断键。
assert grouped[0] == [0, 2, 4]  # 验证defaultdict分组。
