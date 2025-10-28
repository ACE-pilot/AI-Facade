import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import logging

# 配置日志
logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("error.log"),
                              logging.StreamHandler()])

class MAModel(parl.Model):
    def __init__(self,
                 obs_dim,
                 act_dim,
                 obs_shape_n,
                 act_shape_n,
                 continuous_actions=False,
                 discrete_actions=False):
        super(MAModel, self).__init__()
        # 参数校验
        if not isinstance(obs_shape_n, list) or not all(isinstance(shape, tuple) for shape in obs_shape_n):
            raise ValueError("obs_shape_n 必须是包含元组的列表")
        if not isinstance(act_shape_n, list) or not all(isinstance(shape, int) for shape in act_shape_n):
            raise ValueError("act_shape_n 必须是整数的列表")
        # Critic 输入维度
        critic_in_dim = sum([s[0] for s in obs_shape_n]) + sum(act_shape_n)
        logging.info(f"critic_in_dim: {critic_in_dim}")
        # 统一 obs_dim, act_dim
        if isinstance(obs_dim, tuple): obs_dim = obs_dim[0]
        if isinstance(act_dim, tuple): act_dim = act_dim[0]
        # 保存动作类型
        self.continuous_actions = continuous_actions
        self.discrete_actions = discrete_actions
        # 模型定义
        self.actor_model = ActorModel(obs_dim, act_dim, continuous_actions, discrete_actions)
        self.critic_model = CriticModel(critic_in_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, act):
        return self.critic_model(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()

class ActorModel(parl.Model):
    def __init__(self, obs_dim, act_dim, continuous_actions=False, discrete_actions=False):
        super(ActorModel, self).__init__()
        self.continuous_actions = continuous_actions
        self.discrete_actions = discrete_actions
        hid1, hid2 = 64, 64
        # 公共网络层
        self.fc1 = nn.Linear(obs_dim, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        # 输出层
        if self.discrete_actions:
            self.fc_logits = nn.Linear(hid2, act_dim)  # act_dim = n_discrete
        else:
            self.fc3 = nn.Linear(hid2, act_dim)
            if self.continuous_actions:
                self.std_fc = nn.Linear(hid2, act_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        if self.discrete_actions:
            logits = self.fc_logits(x)
            return logits  # 逻辑回归输出
        means = self.fc3(x)
        if self.continuous_actions:
            std = self.std_fc(x)
            std = paddle.maximum(std, paddle.to_tensor(1e-5))
            return means, std
        return means

class CriticModel(parl.Model):
    def __init__(self, critic_in_dim):
        super(CriticModel, self).__init__()
        hid1, hid2, out = 64, 64, 1
        self.fc1 = nn.Linear(critic_in_dim, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, out)

    def forward(self, obs_n, act_n):
        # 拼接输入
        inputs = paddle.concat(obs_n + act_n, axis=1)
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q = self.fc3(x)
        return paddle.squeeze(Q, axis=1)
