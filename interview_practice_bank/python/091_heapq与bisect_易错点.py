"""
题目 091：heapq与bisect_易错点

要求：完成“heapq与bisect”的易错点题，并说明时间复杂度、对象身份或协议行为。
先自己实现，再运行本文件查看参考代码结果。
"""

import bisect  # 导入二分维护有序序列工具。
import heapq  # 导入最小堆工具。
values = [1, 3, 5, 7]  # 创建有序列表。
bisect.insort(values, 5)  # 用二分定位并插入以保持有序。
heap = [8, 3, 6, 1, 9]  # 创建普通列表。
heapq.heapify(heap)  # 原地线性时间构建最小堆。
smallest = heapq.heappop(heap)  # 弹出最小元素。
largest_two = heapq.nlargest(2, heap)  # 高效取得少量最大元素。
assert values == sorted(values) and smallest == 1  # 验证两个数据结构。
print(values, heap, largest_two)  # 输出有序列表和堆。
