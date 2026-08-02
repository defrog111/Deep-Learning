"""
题目 018：sorted的key与稳定排序_变式

要求：完成“sorted的key与稳定排序”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. casefold比lower更适合大小写无关排序。
2. 验证key只参与比较不改变原值。

完成标准：
- 验证key只参与比较不改变原值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

case_insensitive = sorted(['Bob', 'alice', 'ALAN'], key=str.casefold)  # casefold比lower更适合大小写无关排序。
assert case_insensitive == ['ALAN', 'alice', 'Bob']  # 验证key只参与比较不改变原值。
