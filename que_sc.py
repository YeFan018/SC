from collections import deque

class queue:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.queue = deque(maxlen=maxsize)  # 使用deque并设置最大长度

    def put(self, value):
        if len(self.queue) < self.maxsize:
            self.queue.append(value)  # 在队列尾部添加元素
        else:
            self.queue.popleft()  # 如果队列已满，从头部移除一个元素
            self.queue.append(value)  # 然后在尾部添加新元素

    def pop(self, index=0):
        if index == 0:
            self.queue.popleft()  # 从头部移除元素
        else:
            self.queue.pop()  # 从尾部移除元素

    def get(self):
        return list(self.queue)  # 返回队列中的所有元素，转换为列表