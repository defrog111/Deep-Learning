"""
题目 078：collections高频工具_变式

要求：完成“collections高频工具”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入高频容器。
2. 导入分层映射和默认字典。
3. 组合常用collections工具。
4. 验证覆盖和默认值。

完成标准：
- 验证覆盖和默认值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from collections import Counter, defaultdict, deque  # 导入高频容器。
from collections import ChainMap, defaultdict  # 导入分层映射和默认字典。
layers = ChainMap({'debug': True}, {'debug': False, 'port': 8000}); counts = defaultdict(int); counts['x'] += 1  # 组合常用collections工具。
assert layers['debug'] is True and layers['port'] == 8000 and counts['x'] == 1  # 验证覆盖和默认值。
