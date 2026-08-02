"""
题目 068：浅拷贝与深拷贝_综合

要求：完成“浅拷贝与深拷贝”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入复制工具。
2. 导入垃圾回收接口。
3. 检查容器跟踪并主动执行循环垃圾回收。
4. 验证GC接口返回值。

完成标准：
- 验证GC接口返回值。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import copy  # 导入复制工具。
import gc  # 导入垃圾回收接口。
tracked_before = gc.is_tracked([]); collected = gc.collect()  # 检查容器跟踪并主动执行循环垃圾回收。
assert tracked_before and isinstance(collected, int)  # 验证GC接口返回值。
