"""
题目 023：函数参数args与kwargs_易错点

要求：完成“函数参数args与kwargs”的易错点题，并说明时间复杂度、对象身份或协议行为。

操作步骤：
1. 默认列表只创建一次，是常见面试陷阱。
    bucket.append(value)  # 多次调用共享同一个列表。
    return list(bucket)  # 返回副本便于观察。
2. 验证默认参数共享状态。

完成标准：
- 验证默认参数共享状态。
- 脚本能够独立运行，并输出便于人工检查的结果。

练习方式：先只看题目和步骤自己实现，再阅读下面的参考代码。
"""

def mutable_default(value, bucket=[]):  # 默认列表只创建一次，是常见面试陷阱。
    bucket.append(value)  # 多次调用共享同一个列表。
    return list(bucket)  # 返回副本便于观察。
assert mutable_default(1) == [1] and mutable_default(2) == [1, 2]  # 验证默认参数共享状态。
