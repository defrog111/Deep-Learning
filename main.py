class Node:  # 定义双向链表的节点类。
    def __init__(self, key=0, value=0):  # 初始化节点，保存 key 和 value。
        self.key = key  # 记录当前节点的键，删除节点时需要用它从哈希表里移除。
        self.value = value  # 记录当前节点的值，get 的时候要返回它。
        self.prev = None  # 指向前一个节点，方便 O(1) 删除。
        self.next = None  # 指向后一个节点，方便 O(1) 插入。


class LRUCache:  # 定义 LRU 缓存类。
    def __init__(self, capacity: int):  # 初始化缓存容量和辅助结构。
        self.capacity = capacity  # 保存缓存的最大容量。
        self.cache = {}  # 用哈希表记录 key 到节点的映射，保证 O(1) 查找。
        self.left = Node()  # 创建左侧虚拟头节点，left.next 表示最久未使用的真实节点。
        self.right = Node()  # 创建右侧虚拟尾节点，right.prev 表示最近使用的真实节点。
        self.left.next = self.right  # 把头节点的 next 指向尾节点，形成初始空链表。
        self.right.prev = self.left  # 把尾节点的 prev 指向头节点，形成初始空链表。

    def remove(self, node: Node) -> None:  # 把某个节点从双向链表中删除。
        prev_node = node.prev  # 先找到当前节点的前一个节点。
        next_node = node.next  # 再找到当前节点的后一个节点。
        prev_node.next = next_node  # 让前一个节点直接指向后一个节点。
        next_node.prev = prev_node  # 让后一个节点直接指回前一个节点。

    def insert(self, node: Node) -> None:  # 把某个节点插入到链表最右边，表示最近使用。
        prev_node = self.right.prev  # 找到当前链表里最后一个真实节点。
        next_node = self.right  # 右侧虚拟尾节点就是插入位置的后一个节点。
        prev_node.next = node  # 让原来的最后一个节点指向新节点。
        node.prev = prev_node  # 让新节点的 prev 指向原来的最后一个节点。
        node.next = next_node  # 让新节点的 next 指向尾节点。
        next_node.prev = node  # 让尾节点的 prev 指向新节点。

    def get(self, key: int) -> int:  # 读取 key 对应的值。
        if key not in self.cache:  # 如果哈希表里没有这个 key，说明缓存未命中。
            return -1  # 按题目要求返回 -1。

        node = self.cache[key]  # 取出这个 key 对应的链表节点。
        self.remove(node)  # 先把它从原来的位置删掉。
        self.insert(node)  # 再把它插到最右边，表示它刚刚被访问过。
        return node.value  # 返回节点里保存的值。

    def put(self, key: int, value: int) -> None:  # 插入或更新一个 key-value。
        if key in self.cache:  # 如果这个 key 已经存在，说明是更新操作。
            old_node = self.cache[key]  # 先取出旧节点。
            self.remove(old_node)  # 把旧节点从链表里删除。
            del self.cache[key]  # 把旧 key 从哈希表里删掉，准备放入新节点。

        new_node = Node(key, value)  # 创建一个新的节点来保存最新的 key 和 value。
        self.cache[key] = new_node  # 把新节点放进哈希表，方便 O(1) 查找。
        self.insert(new_node)  # 把新节点插到链表最右边，表示最近使用。

        if len(self.cache) > self.capacity:  # 如果当前缓存数量超过容量，就要淘汰最久未使用的节点。
            lru = self.left.next  # 头节点后面的第一个真实节点就是最久未使用的节点。
            self.remove(lru)  # 把这个最久未使用的节点从链表中删除。
            del self.cache[lru.key]  # 再从哈希表中删除对应的 key。


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key, value)
