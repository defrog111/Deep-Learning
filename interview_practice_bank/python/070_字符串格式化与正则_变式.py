"""
题目 070：字符串格式化与正则_变式

要求：完成“字符串格式化与正则”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入正则表达式模块。
2. 预编译带命名组和边界的正则。
3. 验证fullmatch与命名组。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import re  # 导入正则表达式模块。
compiled = re.compile(r'(?P<name>[A-Za-z]+)-(?P<number>\d+)$', flags=re.ASCII)  # 预编译带命名组和边界的正则。
match = compiled.fullmatch('item-42'); assert match and match.groupdict() == {'name': 'item', 'number': '42'}  # 验证fullmatch与命名组。
