"""
题目 100：复杂度与常见陷阱_综合

要求：完成“复杂度与常见陷阱”的综合题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入异步I/O框架。
2. 定义协程函数。
    await asyncio.sleep(0)  # 主动让出事件循环。
    return value * value  # 返回计算结果。
async def async_main():  # 定义异步入口。
    return await asyncio.gather(*(async_square(value) for value in range(4)))  # 并发等待多个协程并保持顺序。
3. 创建事件循环运行异步入口。

完成标准：
- 关键结果的shape、类型或数值符合题目要求。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import asyncio  # 导入异步I/O框架。
async def async_square(value):  # 定义协程函数。
    await asyncio.sleep(0)  # 主动让出事件循环。
    return value * value  # 返回计算结果。
async def async_main():  # 定义异步入口。
    return await asyncio.gather(*(async_square(value) for value in range(4)))  # 并发等待多个协程并保持顺序。
async_result = asyncio.run(async_main()); assert async_result == [0, 1, 4, 9]  # 创建事件循环运行异步入口。
