"""
题目 092：heapq与bisect_综合

要求：完成“heapq与bisect”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建有序列表。
2. 用二分定位并插入以保持有序。
3. 创建普通列表。
4. 原地线性时间构建最小堆。
5. 弹出最小元素。
6. 高效取得少量最大元素。
7. 序号用于优先级相同时保持稳定并避免比较任务对象。
for priority, task in [(1, 'a'), (1, 'b'), (0, 'c')]:  # 加入多个任务。
    heapq.heappush(tasks, (priority, sequence, task)); sequence += 1  # 使用三元组实现稳定优先队列。

完成标准：
- 验证两个数据结构。
- 验证优先级和稳定顺序。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import bisect  # 导入二分维护有序序列工具。
import heapq  # 导入最小堆工具。
values = [1, 3, 5, 7]  # 创建有序列表。
bisect.insort(values, 6)  # 用二分定位并插入以保持有序。
heap = [8, 3, 6, 1, 9]  # 创建普通列表。
heapq.heapify(heap)  # 原地线性时间构建最小堆。
smallest = heapq.heappop(heap)  # 弹出最小元素。
largest_two = heapq.nlargest(2, heap)  # 高效取得少量最大元素。
assert values == sorted(values) and smallest == 1  # 验证两个数据结构。
print(values, heap, largest_two)  # 输出有序列表和堆。
tasks = []; sequence = 0  # 序号用于优先级相同时保持稳定并避免比较任务对象。
for priority, task in [(1, 'a'), (1, 'b'), (0, 'c')]:  # 加入多个任务。
    heapq.heappush(tasks, (priority, sequence, task)); sequence += 1  # 使用三元组实现稳定优先队列。
assert [heapq.heappop(tasks)[2] for _ in range(3)] == ['c', 'a', 'b']  # 验证优先级和稳定顺序。
