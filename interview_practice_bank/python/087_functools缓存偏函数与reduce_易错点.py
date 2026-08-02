"""
题目 087：functools缓存偏函数与reduce_易错点

要求：完成“functools缓存偏函数与reduce”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入函数式工具。
2. 导入LRU缓存。
3. 限制缓存容量防止无界增长。
def square(value):  # 定义纯函数。
    return value * value  # 计算平方。
4. 验证命中统计。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from functools import cache, partial, reduce  # 导入函数式工具。
from functools import lru_cache  # 导入LRU缓存。
@lru_cache(maxsize=2)  # 限制缓存容量防止无界增长。
def square(value):  # 定义纯函数。
    return value * value  # 计算平方。
square(1); square(1); assert square.cache_info().hits == 1  # 验证命中统计。
