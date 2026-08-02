"""
题目 091：heapq与bisect_易错点

要求：完成“heapq与bisect”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入二分维护有序序列工具。
2. 导入最小堆工具。
3. 导入二分工具。
4. 区分重复值左右插入点。
5. 验证闭开区间边界。

完成标准：
- 验证闭开区间边界。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import bisect  # 导入二分维护有序序列工具。
import heapq  # 导入最小堆工具。
import bisect  # 导入二分工具。
ordered = [1, 2, 2, 4]; left_position = bisect.bisect_left(ordered, 2); right_position = bisect.bisect_right(ordered, 2)  # 区分重复值左右插入点。
assert (left_position, right_position) == (1, 3)  # 验证闭开区间边界。
