"""
题目 012：字典集合与哈希_综合

要求：完成“字典集合与哈希”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入多层配置查找结构。
2. 前层配置覆盖后层默认值。
3. 验证ChainMap查找顺序。

完成标准：
- 验证ChainMap查找顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import ChainMap  # 导入多层配置查找结构。
settings = ChainMap({'timeout': 10}, {'timeout': 30, 'retries': 3})  # 前层配置覆盖后层默认值。
assert settings['timeout'] == 10 and settings['retries'] == 3  # 验证ChainMap查找顺序。
