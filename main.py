from collections import Counter  # 导入 Counter，用来统计每个数字出现的次数，返回的是映射对象，不是张量，没有 shape。
from typing import List  # 导入 List 类型注解，用来标明输入和输出都是 Python 列表，不是张量，没有 shape。


class Solution:  # 定义题解类，这里不是张量，没有 shape。
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:  # 定义求前 k 个高频元素的方法，nums 的长度是 len(nums)，不是张量 shape。
        counts = Counter(nums)  # 统计每个数字出现的频次，counts 的键是数字、值是频次，不是张量，没有 shape。
        buckets = [[] for _ in range(len(nums) + 1)]  # 创建桶数组，最外层列表长度是 len(nums) + 1，每个桶里放同频次的数字。

        for num, freq in counts.items():  # 遍历每个数字和它的频次，这里 num 和 freq 都是标量，不是张量，没有 shape。
            buckets[freq].append(num)  # 把当前数字放进对应频次的桶里，buckets[freq] 是一个 Python 列表，没有张量 shape。

        result = []  # 初始化答案列表，长度会从 0 逐步增长到 k。
        for freq in range(len(buckets) - 1, 0, -1):  # 从高频桶往低频桶倒着遍历，这里 freq 是整数，没有 shape。
            for num in buckets[freq]:  # 遍历当前频次桶中的所有数字，这里 num 是整数，没有 shape。
                result.append(num)  # 把当前数字追加到答案列表中，result 的长度加 1。
                if len(result) == k:  # 如果答案列表长度已经达到 k，就可以提前结束，这里是布尔判断，没有 shape。
                    return result  # 返回前 k 个高频元素组成的列表，长度是 k，不是张量，没有 shape。

        return result  # 如果循环自然结束，就返回答案列表，长度至多是不同数字的个数，不是张量，没有 shape。
