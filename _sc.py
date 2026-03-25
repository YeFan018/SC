import numpy as np
import torch
import random  # ✅ 如果没有这行请添加
import os  # ✅ 如果没有这行请添加
from env_sc import env
from replay_memory_sc import ReplayMemory
from sac_sc import SAC
from utils_sc import load_semantic_matrix, combine_agents_by_reward_and_semantic, distribute_agents
from torch.utils.tensorboard import SummaryWriter
import datetime


# ✅ 新增：设置固定种子函数
def set_global_seed(seed=42):
    """设置全局随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# 环境和代理的参数（保持不变）
content_n = 50
request_t = 50
edge_n = 4
end_edge = 10
edge_size = 10
episodes = 1000

# 加载语义相似度矩阵（保持不变）
semantic_matrix = load_semantic_matrix(r'semantic_matrix.npy')
if semantic_matrix is None:
    raise ValueError("无法加载语义矩阵，终止程序")


class Args:
    def __init__(self):
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.4
        self.policy = 'GaussianPolicy_noLSTM'
        self.target_update_interval = 1
        self.automatic_entropy_tuning = True
        self.lr = 0.0003
        self.cuda = True
        self.hidden_size = 256
        self.batch_size = 128


def zipf(content_n, end_n, table, t_request, a=1):
    p = np.array([1 / (i ** a) for i in range(1, content_n + 1)])
    p = p / sum(p)
    request = np.zeros((end_n, t_request))
    for i in range(end_n):
        for j in range(t_request):
            c = np.random.choice(list(range(len(table[i]))), 1, False, p)
            c = c[0]
            request[i, j] = table[i][c]
    return request


def main():
    set_global_seed(39)  # 您可以改为任何固定数字

    args = Args()

    # 初始化环境（完全不变）
    caching_env = env(edge_n, end_edge, edge_size, content_n)
    # 创建代理（完全不变）
    agent_list = [SAC(caching_env.state_n, caching_env.action_n, args, content_n=content_n) for _ in range(edge_n)]
    memory_list = [ReplayMemory(100000, seed=1234) for _ in range(edge_n)]

    # 学习率调度器（完全不变）
    lr_scheduler = [torch.optim.lr_scheduler.ExponentialLR(agent.policy_optim, gamma=0.99) for agent in agent_list]
    lr_scheduler_critic = [torch.optim.lr_scheduler.ExponentialLR(agent.critic_optim, gamma=0.99) for agent in
                           agent_list]

    # TensorBoard 写入器（完全不变）
    writer = SummaryWriter(f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_semantic_cache')

    # 请求初始化（完全不变）
    table = np.array([np.random.permutation(list(range(content_n))) for _ in range(edge_n * end_edge)])
    requests = zipf(content_n, edge_n * end_edge, table, request_t + 1).astype(int)
    requests = requests.reshape(edge_n, end_edge, request_t + 1)

    RL_step = [0] * edge_n

    # 训练循环（完全不变）
    for episode in range(1, episodes + 1):
        caching_env.reset()

        episode_reward_sum = 0
        episode_latency_sum = 0
        episode_hit_rate_sum = 0
        episode_semantic_hit_rate_sum = 0
        edge_semantic_hits_list = [0] * edge_n

        for step in range(request_t):
            done = (step == request_t - 1)

            cur_state_list = []
            action_list = []
            cur_requests = []

            for agent in range(edge_n):
                cur_reqs = [requests[agent][u][step] for u in range(end_edge)]
                cur_requests.append(cur_reqs)

                edge_action_list = []
                edge_state_list = []

                for user in range(end_edge):
                    user_state = caching_env.transform_state(caching_env.cache_state[agent], [cur_reqs[user]],
                                                             semantic_matrix)
                    edge_state_list.append(user_state)
                    action = agent_list[agent].select_action(user_state, semantic_matrix=semantic_matrix)
                    edge_action_list.append(action)

                    # 计算语义命中
                    if semantic_matrix[action][cur_reqs[user]] > 0.7 and action != cur_reqs[user]:
                        edge_semantic_hits_list[agent] += 1

                cur_state_list.append(edge_state_list)
                action_list.append(edge_action_list)

            # 执行动作
            reward, latency, hit_rate, semantic_hit_rate = caching_env.step(action_list, cur_requests, semantic_matrix)
            episode_reward_sum += reward
            episode_latency_sum += latency
            episode_hit_rate_sum += hit_rate
            episode_semantic_hit_rate_sum += semantic_hit_rate

            # 存经验 & 更新参数
            for agent in range(edge_n):
                for user in range(end_edge):
                    next_req = requests[agent][user][step + 1] if step < request_t else requests[agent][user][step]
                    next_state = caching_env.transform_state(caching_env.cache_state[agent], [next_req],
                                                             semantic_matrix)
                    memory_list[agent].push(
                        cur_state_list[agent][user],
                        action_list[agent][user],
                        caching_env.cur_reward[agent],
                        next_state,
                        done,
                        cur_state_list[agent][user],
                        next_state
                    )

                if len(memory_list[agent]) > args.batch_size:
                    losses = agent_list[agent].update_parameters(memory_list[agent], args.batch_size, RL_step[agent])
                    if losses is not None:
                        critic_loss, policy_loss = losses
                        RL_step[agent] += 1

        # 联邦同步（完全不变）
        if episode % 50 == 0:
            scores = caching_env.reward.copy()
            # 计算每个代理的语义命中率
            semantic_rates = [edge_semantic_hits / (end_edge * request_t) for edge_semantic_hits in
                              edge_semantic_hits_list]
            cen_agent = combine_agents_by_reward_and_semantic(
                main_agent=agent_list[0],
                agents=agent_list,
                scores=scores,
                semantic_matrix=semantic_matrix,
                semantic_rates=semantic_rates,
                main_agent_idx=0,
                dynamic=True,
                base_threshold=0.1,
                tau_critic=0.005,
                tau_policy=0.1
            )
            agent_list = distribute_agents(cen_agent, agent_list)
            print(f"[轮次 {episode}] Federated update with semantic enhancement performed.")

        # 学习率衰减（完全不变）
        if episode > 100:
            for scheduler in lr_scheduler:
                scheduler.step()
            for scheduler in lr_scheduler_critic:
                scheduler.step()

        # 计算平均值（完全不变）
        avg_reward = episode_reward_sum / request_t
        avg_latency = episode_latency_sum / request_t
        avg_hit_rate = episode_hit_rate_sum / request_t
        avg_semantic_hit_rate = episode_semantic_hit_rate_sum / request_t

        print(
            f"[轮次 {episode}] Avg Reward: {avg_reward:.4f}, Avg Latency: {avg_latency:.4f}, Avg Hit Rate: {avg_hit_rate:.4f}, Avg Semantic Hit Rate: {avg_semantic_hit_rate:.4f}")
        writer.add_scalar('Reward', avg_reward, episode)
        writer.add_scalar('Latency', avg_latency, episode)
        writer.add_scalar('Hit Rate', avg_hit_rate, episode)
        writer.add_scalar('Semantic Hit Rate', avg_semantic_hit_rate, episode)


if __name__ == "__main__":
    main()