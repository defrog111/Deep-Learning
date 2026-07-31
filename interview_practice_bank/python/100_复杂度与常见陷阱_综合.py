"""
题目 100：复杂度与常见陷阱_综合

要求：完成“复杂度与常见陷阱”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. list顺序存储且按位置访问O(1)。
2. set平均O(1)成员测试但需额外内存。
3. 设置查找目标。
4. list成员测试最坏O(n)。
5. set成员测试平均O(1)。
6. 重复同一个内部列表引用的常见陷阱。
7. 每行创建独立列表。
8. 修改一行会影响所有共享行。
9. 正确矩阵只修改第一行。
10. 定义协程函数。
    await asyncio.sleep(0)  # 主动让出事件循环。
    return value * value  # 返回计算结果。
async def async_main():  # 定义异步入口。
    return await asyncio.gather(*(async_square(value) for value in range(4)))  # 并发等待多个协程并保持顺序。

完成标准：
- 验证复杂度容器和别名陷阱。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

values = list(range(1000))  # list顺序存储且按位置访问O(1)。
lookup = set(values)  # set平均O(1)成员测试但需额外内存。
target = 60  # 设置查找目标。
in_list = target in values  # list成员测试最坏O(n)。
in_set = target in lookup  # set成员测试平均O(1)。
matrix_bad = [[0] * 3] * 3  # 重复同一个内部列表引用的常见陷阱。
matrix_good = [[0 for _ in range(3)] for _ in range(3)]  # 每行创建独立列表。
matrix_bad[0][0] = 1  # 修改一行会影响所有共享行。
matrix_good[0][0] = 1  # 正确矩阵只修改第一行。
assert in_list == in_set and sum(row[0] for row in matrix_bad) == 3  # 验证复杂度容器和别名陷阱。
print(matrix_bad, matrix_good)  # 输出错误与正确矩阵。
import asyncio  # 导入异步I/O框架。
async def async_square(value):  # 定义协程函数。
    await asyncio.sleep(0)  # 主动让出事件循环。
    return value * value  # 返回计算结果。
async def async_main():  # 定义异步入口。
    return await asyncio.gather(*(async_square(value) for value in range(4)))  # 并发等待多个协程并保持顺序。
async_result = asyncio.run(async_main()); assert async_result == [0, 1, 4, 9]  # 创建事件循环运行异步入口。
