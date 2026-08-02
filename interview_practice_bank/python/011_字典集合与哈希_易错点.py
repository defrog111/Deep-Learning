"""
题目 011：字典集合与哈希_易错点

要求：完成“字典集合与哈希”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 演示不可哈希对象不能作为set元素。
    {['bad']}  # list可变因此不可哈希。
except TypeError as error:  # 捕获预期错误。
    unhashable_message = str(error)  # 保存错误信息。
2. 验证哈希约束。

完成标准：
- 验证哈希约束。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

try:  # 演示不可哈希对象不能作为set元素。
    {['bad']}  # list可变因此不可哈希。
except TypeError as error:  # 捕获预期错误。
    unhashable_message = str(error)  # 保存错误信息。
assert 'unhashable' in unhashable_message  # 验证哈希约束。
