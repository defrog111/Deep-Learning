"""
题目 099：复杂度与常见陷阱_易错点

要求：完成“复杂度与常见陷阱”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 创建锁和共享资源。
with lock:  # 对复合共享操作加锁；GIL不等于业务操作原子性。
    shared.append('safe')  # 在临界区修改数据。
2. 退出with后锁已释放。
3. 释放验证时重新取得的锁。

完成标准：
- 退出with后锁已释放。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

import threading  # 导入线程同步原语。
lock = threading.Lock(); shared = []  # 创建锁和共享资源。
with lock:  # 对复合共享操作加锁；GIL不等于业务操作原子性。
    shared.append('safe')  # 在临界区修改数据。
assert shared == ['safe'] and lock.acquire(blocking=False)  # 退出with后锁已释放。
lock.release()  # 释放验证时重新取得的锁。
