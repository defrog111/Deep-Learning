"""
题目 090：heapq与bisect_变式

要求：完成“heapq与bisect”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入二分维护有序序列工具。
2. 导入最小堆工具。
3. 导入堆工具。
4. 用元组建立最小优先队列。
5. 验证按优先级弹出。

完成标准：
- 验证按优先级弹出。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import bisect  # 导入二分维护有序序列工具。
import heapq  # 导入最小堆工具。
import heapq  # 导入堆工具。
priority_queue = []; heapq.heappush(priority_queue, (2, 'low')); heapq.heappush(priority_queue, (1, 'high')); first_priority = heapq.heappop(priority_queue)  # 用元组建立最小优先队列。
assert first_priority == (1, 'high')  # 验证按优先级弹出。
