"""
题目 070：字符串格式化与正则_变式

要求：完成“字符串格式化与正则”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 构造待解析文本。
2. 使用命名组提取字段。
3. 处理匹配失败情况。
4. 明确抛出解析错误。
5. 把命名组转换为字典。
6. 使用格式说明符对齐和补零。
7. 预编译带命名组和边界的正则。

完成标准：
- 验证正则提取。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import re  # 导入正则表达式模块。
text = 'User Alice email alice@example.com score 95'  # 构造待解析文本。
match = re.search(r'(?P<name>[A-Z][a-z]+).*?(?P<email>[\w.]+@[\w.]+).*?(?P<score>\d+)', text)  # 使用命名组提取字段。
if match is None:  # 处理匹配失败情况。
    raise ValueError('pattern did not match')  # 明确抛出解析错误。
fields = match.groupdict()  # 把命名组转换为字典。
message = '{:<10} | {:04d}'.format(fields['name'], int(fields['score']))  # 使用格式说明符对齐和补零。
assert fields['email'] == 'alice@example.com'  # 验证正则提取。
print(fields, message)  # 输出解析和格式化结果。
compiled = re.compile(r'(?P<name>[A-Za-z]+)-(?P<number>\d+)$', flags=re.ASCII)  # 预编译带命名组和边界的正则。
match = compiled.fullmatch('item-42'); assert match and match.groupdict() == {'name': 'item', 'number': '42'}  # 验证fullmatch与命名组。
