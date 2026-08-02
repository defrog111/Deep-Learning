"""
题目 098：复杂度与常见陷阱_变式

要求：完成“复杂度与常见陷阱”的变式题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 导入线程池。
2. 线程适合I/O等待型任务。
    threaded = list(executor.map(lambda value: value * value, range(5)))  # 保持输入顺序收集结果。
3. 验证线程池接口。

完成标准：
- 验证线程池接口。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

from concurrent.futures import ThreadPoolExecutor  # 导入线程池。
with ThreadPoolExecutor(max_workers=2) as executor:  # 线程适合I/O等待型任务。
    threaded = list(executor.map(lambda value: value * value, range(5)))  # 保持输入顺序收集结果。
assert threaded == [0, 1, 4, 9, 16]  # 验证线程池接口。
