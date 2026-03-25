import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils_sc import soft_update, hard_update
from model_sc import GaussianPolicy, QNetwork, GaussianPolicy_noLSTM

class SAC(object):
    def __init__(self, num_inputs, action_space, args, content_n=None):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.critic = QNetwork(num_inputs, action_space, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_inputs, action_space, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)
        if self.automatic_entropy_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(action_space).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        self.policy = GaussianPolicy_noLSTM(num_inputs, action_space, args.hidden_size, content_n=content_n).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, evaluate=False, semantic_matrix=None):
        if state is None or not isinstance(state, (np.ndarray, list)) or len(state) == 0:
            raise ValueError("无效的状态输入")
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        action, (action_probabilities, log_action_probabilities) = self.policy.sample(state, semantic_matrix)
        entropy = -torch.sum(action_probabilities * log_action_probabilities, dim=1).mean().item()
        #print(f"Action entropy: {entropy}")
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, memory, batch_size, updates):
        """
        更新网络参数
        """
        # 从经验回放中采样一个批次
        batch = memory.sample(batch_size)
        if batch is None:
            print("无法进行训练：无效的回放数据")
            return


        # 解包经验数据
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch, semantic_state, next_semantic_state = batch
        # 安全转换所有批量张量
        def to_tensor(x, dtype=torch.float):
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=dtype)
            return x.to(self.device)

        state_batch = to_tensor(state_batch)
        next_state_batch = to_tensor(next_state_batch)
        action_batch = to_tensor(action_batch, dtype=torch.long).view(-1, 1)
        reward_batch = to_tensor(reward_batch).view(-1, 1)
        mask_batch = to_tensor(mask_batch).view(-1, 1)

        with torch.no_grad():
            next_state_action, (action_probabilities, log_action_probabilities) = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
            min_qf_next_target = action_probabilities * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1, keepdim=True)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.gamma * min_qf_next_target

        # 计算当前 Q 值的损失
        qf1_, qf2_ = self.critic(state_batch)
        qf1 = qf1_.gather(1, action_batch)
        qf2 = qf2_.gather(1, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # 更新策略网络
        _, (action_probabilities, log_action_probabilities) = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        return qf_loss.item(), policy_loss.item()

    def parameters(self):
        return list(self.critic.parameters()) \
            + list(self.critic_target.parameters()) \
            + list(self.policy.parameters())

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        """
        保存模型参数
        """
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        """
        加载模型参数
        """
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
