"""
题目 072：字符串格式化与正则_综合

要求：完成“字符串格式化与正则”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 构造待解析文本。
2. 使用命名组提取字段。
3. 处理匹配失败情况。
4. 明确抛出解析错误。
5. 把命名组转换为字典。
6. 使用格式说明符对齐和补零。
7. 使用前后查看遮挡手机号中间数字。
8. 综合正则替换和格式说明符。

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
message = '{:<10} | {:06d}'.format(fields['name'], int(fields['score']))  # 使用格式说明符对齐和补零。
assert fields['email'] == 'alice@example.com'  # 验证正则提取。
print(fields, message)  # 输出解析和格式化结果。
redacted = re.sub(r'(?<=\d{3})\d(?=\d{4})', '*', '13812345678')  # 使用前后查看遮挡手机号中间数字。
formatted = f'{1234.5:,.2f}'; assert redacted == '138****5678' and formatted == '1,234.50'  # 综合正则替换和格式说明符。
