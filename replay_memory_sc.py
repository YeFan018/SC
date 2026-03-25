import random
import numpy as np


class ReplayMemory:
    def __init__(self, capacity, seed):
        self.rng = random.Random(seed)  # 替代: random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, semantic_state=None, next_semantic_state=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # 存储时将语义信息与状态一起存储
        self.buffer[self.position] = (state, action, reward, next_state, done, semantic_state, next_semantic_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            print("警告：经验回放缓冲区数据不足，无法采样")
            return None  # 或者返回一个默认值

        batch = self.rng.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, semantic_state, next_semantic_state = map(np.stack, zip(*batch))

        # 返回结果中包括语义信息
        return state, action, reward, next_state, done, semantic_state, next_semantic_state

    def __len__(self):
        return len(self.buffer)
