import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
LOG_SIG_MIN = -20
LOG_SIG_MAX = 2



# 初始化权重函数
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):  # 包括 Q 网络（QNetwork）
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)  # 输入为 num_inputs
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs, hidden_dim)  # 输入为 num_inputs
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)

        x2 = F.relu(self.linear5(state))
        x2 = F.relu(self.linear6(x2))
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)

        return x1, x2

class GaussianPolicy_noLSTM(nn.Module):
    """
    离散动作策略网络（无 LSTM），并加入语义注意力融合。
    输入 state 包含 cache 计数、request 计数和 semantic_state（三段拼接）。
    semantic_matrix 用于在推理时计算当前请求的语义得分。
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, content_n, action_space=None):
        super(GaussianPolicy_noLSTM, self).__init__()
        self.content_n = content_n
        self.num_actions = num_actions
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state, semantic_matrix=None):
        # state: Tensor shape [B, num_inputs]
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        action_logits = self.mean_linear(x)  # [B, num_actions]

        # 仅在单样本推理时加入语义注意力
        if semantic_matrix is not None and state.shape[0] == 1:
            # 提取当前请求在 state 中的位置
            request_idx = state[:, self.content_n:2 * self.content_n].argmax(dim=1).cpu().numpy()[0]
            # 计算 semantic_scores
            semantic_scores = torch.tensor(
                [semantic_matrix[i, request_idx] for i in range(self.num_actions)],
                device=action_logits.device, dtype=torch.float32
            )  # [num_actions]

            # Softmax 归一化
            semantic_weights = F.softmax(semantic_scores, dim=0).unsqueeze(0)  # [1, num_actions]

            # 注意力机制融合示例
            # Q作为 query，K作为 key
            q = semantic_weights.unsqueeze(2)        # [1, N, 1]
            k = action_logits.unsqueeze(1)           # [1, 1, N]
            attn_scores = torch.bmm(q, k)            # [1, N, N]
            attn_weights = F.softmax(attn_scores.sum(dim=1), dim=1)  # [1, N]

            # 最终融合：70% 原始 logits + 30% 注意力调制
            action_logits = 0.7 * action_logits + 0.3 * (attn_weights * action_logits)

        # 最终输出概率分布
        action_probabilities = F.softmax(action_logits, dim=1)
        return action_probabilities

    def sample(self, state, semantic_matrix=None):
        # 生成动作及其对数概率
        action_probabilities = self.forward(state, semantic_matrix)
        dist = Categorical(action_probabilities)
        action = dist.sample()  # [B]
        # 防止 log(0)
        eps_mask = (action_probabilities == 0.0).float() * 1e-6
        log_probs = torch.log(action_probabilities + eps_mask)
        return action, (action_probabilities, log_probs)



# 用于连续动作的策略网络
class GaussianPolicy_orig(nn.Module):  # 连续动作策略（高斯分布）
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_orig, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.0)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


# 用于离散动作的策略网络
class GaussianPolicy_one(nn.Module):  # 离散动作策略（softmax 分布）
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_one, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.Softmax = nn.Softmax(dim=1)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        log_std = self.log_std_linear(x)
        softmax = self.Softmax(torch.clamp_max(log_std, 10))
        log_std = torch.log(1e-8 + softmax)
        return log_std

    def sample(self, state):
        log_std = self.forward(state)
        tmp = RelaxedOneHotCategorical(temperature=0.5, logits=log_std)
        action_sample = tmp.rsample()
        index = torch.argmax(action_sample, dim=1)
        logits_sample = log_std[(range(index.shape[0]), index)].view(-1, 1)
        entropies = logits_sample.sum(dim=1, keepdim=True)
        return action_sample, entropies, None


# 带 LSTM 的策略网络
class GaussianPolicy(nn.Module):  # 带 LSTM 版本
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state, hidden_in):
        """
        :param state: 3-d for lstm
        :param hidden_in: hidden state of lstm
        :return:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = torch.unsqueeze(x, dim=0)  # more dimension
        x, hidden_lstm = self.lstm(x, hidden_in)
        x = torch.squeeze(x, dim=0)  # less dimension

        probability = F.softmax(self.mean_linear(x), dim=1)
        return probability, hidden_lstm

    def sample(self, state, hidden_in):
        action_probabilities, hidden_lstm = self.forward(state, hidden_in)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), hidden_lstm
