import numpy as np
from que_sc import queue

class env:
    def __init__(self, edge_n, end_edge, edge_size, content_n):
        self.edge_n = edge_n
        self.end_edge = end_edge
        self.end_n = edge_n * end_edge
        self.edge_size = edge_size
        self.content_n = content_n
        self.cache_state = [queue(maxsize=self.edge_size) for _ in range(self.edge_n)]
        self.reward = [0] * edge_n
        self.cur_reward = [0] * edge_n
        self.cost = [0] * edge_n
        self.cur_cost = [0] * edge_n
        self.l_cost = [0] * self.edge_n
        self.s_cost = [0] * self.edge_n
        self.h_cost = [0] * self.edge_n

        self.state_n = 4 * content_n
        self.action_n = content_n + 1
        self.v_c = 2000  # 缓存速度高，模拟低时延
        self.v_e = 100   # 边缘速度低，模拟高时延
        self.content_size = 20

        self.requests = []

    def reset(self):
        self.cost = [0] * self.edge_n
        self.cur_cost = [0] * self.edge_n
        self.reward = [0] * self.edge_n
        self.cur_reward = [0] * self.edge_n
        self.cache_state = [queue(maxsize=self.edge_size) for _ in range(self.edge_n)]
        self.l_cost = [0] * self.edge_n
        self.s_cost = [0] * self.edge_n
        self.h_cost = [0] * self.edge_n
        return np.zeros(self.state_n)

    def transform_state(self, cache_state, request, semantic_info):
        part_size = max(1, int(self.state_n / 4))
        c_state = np.zeros(part_size)
        r_state = np.zeros(part_size)

        for i in cache_state.get():
            if isinstance(i, (int, np.integer)) and 0 < i <= self.content_n:
                c_state[i - 1] += 1

        for i in request:
            if isinstance(i, (int, np.integer)) and 0 < i <= self.content_n:
                r_state[i - 1] += 1

        semantic_state = self.get_semantic_state(cache_state, request, semantic_info)

        # 只把前两段转为 int，语义部分保留 float
        cr_int = np.hstack((c_state, r_state)).astype(int)
        return np.hstack((cr_int, semantic_state))

    def get_semantic_state(self, cache_state, request, semantic_info):
        semantic_state = np.zeros(int(self.state_n / 2))
        for content in cache_state.get():
            if not isinstance(content, (int, np.integer)) or content < 0 or content >= self.content_n:
                continue  # 跳过无效内容
            for req in request:
                if not isinstance(req, (int, np.integer)) or req < 0 or req >= self.content_n:
                    continue  # 跳过无效请求
                sim = self.get_semantic_similarity(content, req, semantic_info)
                semantic_state[content] += sim
        return semantic_state

    def get_semantic_similarity(self, content, request, semantic_info):
        return semantic_info[content][request]

    def step(self, action_list, request_list, semantic_info):
        self.cur_reward = [0] * self.edge_n
        self.cur_cost = [0] * self.edge_n
        self.hits = 0  # 缓存命中计数器（包含精确和语义命中）
        self.semantic_hits = 0  # 语义命中计数器

        beta = 1  # 延迟惩罚系数

        # 检查输入维度
        if not (len(action_list) == self.edge_n and len(request_list) == self.edge_n):
            raise ValueError("action_list or request_list length does not match edge_n")

        for edge in range(self.edge_n):
            if not (len(action_list[edge]) == self.end_edge and len(request_list[edge]) == self.end_edge):
                raise ValueError(f"Invalid action_list or request_list length for edge {edge}")

            for user in range(self.end_edge):
                action = action_list[edge][user]
                request = request_list[edge][user]

                # 检查 semantic_info 索引
                if action >= semantic_info.shape[0] or request >= semantic_info.shape[1]:
                    sim = 0.0
                else:
                    sim = semantic_info[action][request]

                # 计算命中和成本
                if action == request:
                    self.hits += 1  # 精确命中
                    cost = self.content_size / self.v_c  # 0.01（模拟10ms）
                    base_reward = 1.8
                elif sim > 0.8:  # 语义命中
                    self.hits += 1
                    self.semantic_hits += 1
                    cost = self.content_size / self.v_c   # 0.01（模拟10ms）
                    base_reward = 0.6 + 0.2 * sim
                else:
                    cost = self.content_size / self.v_e  # 0.2（模拟200ms）
                    base_reward = max(0.3, 0.2 * sim)

                reward = max(base_reward - beta * cost, 0.0)
                self.cur_cost[edge] += cost
                self.cur_reward[edge] += reward

        self.reward = self.cur_reward
        self.cost = self.cur_cost

        # 计算命中率
        total_requests = self.edge_n * self.end_edge
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        semantic_hit_rate = self.semantic_hits / total_requests if total_requests > 0 else 0.0

        return sum(self.cur_reward) / self.edge_n, sum(self.cur_cost) / self.edge_n, hit_rate, semantic_hit_rate