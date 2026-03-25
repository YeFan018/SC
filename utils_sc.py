import numpy as np
import torch


# 语义相似度矩阵加载
def load_semantic_matrix(filepath):
    try:
        matrix = np.load(filepath)
        print("语义矩阵加载成功")
        return matrix
    except Exception as e:
        print(f"语义矩阵加载失败: {e}")
        return None


# 语义相似度计算
def compute_semantic_similarity(agent_idx, main_idx, semantic_matrix):
    sim = semantic_matrix[agent_idx][main_idx]
    print(f"[Debug] 相似度 (Agent {agent_idx}, Main {main_idx}): sim={sim:.4f}")
    return sim


def combine_agents_by_reward_and_semantic(
    main_agent,
    agents,
    scores,
    semantic_matrix,
    semantic_rates,
    main_agent_idx=0,
    dynamic=True,
    base_threshold=0.1,
    tau_critic=0.005,
    tau_policy=0.1  #*
):

    # 计算 scores 归一化后的逆数权重
    scores = np.array(scores, dtype=float)
    if np.max(scores) == np.min(scores):
        norm_scores = np.ones_like(scores)
    else:
        norm_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
    inv_scores = 1.0 / (norm_scores + 1e-8)

    # 计算语义相似度
    agent_num = len(agents)
    sims = np.array([compute_semantic_similarity(i, main_agent_idx, semantic_matrix) for i in range(agent_num)])

    # 动态或固定阈值
    if dynamic:
        sim_threshold = 0.4
    else:
        sim_threshold = base_threshold

    critic_param_dict = {}
    policy_param_dict = {}
    total_weight = 0.0
    valid_found = False

    # 聚合参数
    for i, agent in enumerate(agents):
        sim = sims[i]
        reward_weight = inv_scores[i]
        sem_rate = semantic_rates[i]
        # α 权重，这里设为 1.0
        weight = sim * (reward_weight + sem_rate)
        if sim < sim_threshold:
            continue
        valid_found = True
        total_weight += weight
        # 累积 critic 模型参数
        for name, param in agent.critic.named_parameters():
            critic_param_dict.setdefault(name, param.data.clone() * weight)
            critic_param_dict[name] += param.data * weight
        # 累积 policy 模型参数
        for name, param in agent.policy.named_parameters():
            policy_param_dict.setdefault(name, param.data.clone() * weight)
            policy_param_dict[name] += param.data * weight

    if not valid_found or total_weight == 0:
        # 回退到 FedAvg
        return combine_agents(main_agent, agents)

    # 更新主模型 critic
    for name, param in main_agent.critic.named_parameters():
        aggregated = critic_param_dict[name] / total_weight
        param.data.copy_(param.data * (1 - tau_critic) + aggregated * tau_critic)
    # 更新主模型 policy
    for name, param in main_agent.policy.named_parameters():
        aggregated = policy_param_dict[name] / total_weight
        param.data.copy_(param.data * (1 - tau_policy) + aggregated * tau_policy)

    return main_agent



# Soft Update
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


# Hard Update
def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(source_param.data)

# 联邦平均
def combine_agents(cen_agent, agent_list):
    for param_cen, _ in zip(cen_agent.parameters(), agent_list[0].parameters()):
        param_cen.data.zero_()

    for agent in agent_list:
        for param_cen, param in zip(cen_agent.parameters(), agent.parameters()):
            param_cen.data.add_(param.data)

    for param_cen in cen_agent.parameters():
        param_cen.data.div_(len(agent_list))

    return cen_agent


# 联邦分发
def distribute_agents(cen_agent, agent_list):
    for agent in agent_list:
        for param_cen, param in zip(cen_agent.parameters(), agent.parameters()):
            param.data.copy_(param_cen.data)
    return agent_list