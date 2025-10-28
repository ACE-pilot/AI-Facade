# mappo.py
import paddle
import warnings
warnings.simplefilter('default')

import numpy as np
from copy import deepcopy
from paddle import fluid
from parl.core.fluid import layers
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid.policy_distribution import SoftCategoricalDistribution, SoftMultiCategoricalDistribution
from gym import spaces  # <–– 新增

__all__ = ['MAPPO']

def SoftPDistribution(logits, act_space):
    """根据动作空间返回对应分布：Discrete/ MultiDiscrete/ Box->Gaussian"""
    # 离散动作
    if hasattr(act_space, 'n'):
        return SoftCategoricalDistribution(logits)
    # MultiDiscrete
    if hasattr(act_space, 'num_discrete_space'):
        return SoftMultiCategoricalDistribution(logits, act_space.low, act_space.high)
    # 连续动作：Gaussian
    if isinstance(act_space, spaces.Box):
        # logits 应该是 (means, std)
        means, std = logits
        return GaussianDistribution(means, std)
    # 其他类型
    raise AssertionError(f"Unsupported action space: {act_space}")

class GaussianDistribution:
    """一个简单的 Gaussian 分布：sample + log_prob"""
    def __init__(self, means, std):
        self.means = means
        self.std = std

    def sample(self):
        # 在 Paddle 中，用 randn 生成 N(0,1)
        noise = paddle.randn(self.means.shape)
        # 如果需要，cast 到和 means 相同的 dtype
        if noise.dtype != self.means.dtype:
            noise = paddle.cast(noise, self.means.dtype)
        return self.means + noise * self.std

    def log_prob(self, actions):
        var = layers.square(self.std)
        term1 = -0.5 * layers.elementwise_div(
            layers.square(actions - self.means), var, axis=1
        )
        log_scale = 0.5 * layers.log(2.0 * np.pi * var)
        return layers.reduce_sum(term1 - log_scale, axis=1)

class MAPPO(Algorithm):
    """
    Multi-Agent PPO (MAPPO) with centralized critic and decentralized actors.
    支持离散 & 连续动作空间。
    """
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=0.99,
                 policy_lr=1e-4,
                 value_lr=1e-3,
                 epsilon=0.2):
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        self.model = model
        self.old_policy = deepcopy(model.actor_model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.epsilon = epsilon

    def predict(self, obs):
        # actor 选动作
        logits = self.model.policy(obs)
        dist = SoftPDistribution(logits, self.act_space[self.agent_index])
        return dist.sample()

    def sample(self, obs):
        # 同 predict
        return self.predict(obs)

    def Q(self, obs_n, act_n):
        # 中央 critic
        return self.model.value(obs_n, act_n)

    def learn(self, obs_n, act_n, rewards, next_obs_n, dones):
        # Critic 更新
        V = self.model.value(obs_n, act_n)
        # next 动作用 old_policy
        next_act_n = []
        for i, obs in enumerate(next_obs_n):
            logits = self.old_policy.forward(obs)
            dist = SoftPDistribution(logits, self.act_space[i])
            next_act_n.append(dist.sample())
        V_next = self.model.value(next_obs_n, next_act_n)
        # TD(0) 目标
        target_V = rewards + self.gamma * (1 - dones) * V_next
        td_error = layers.square_error_cost(V, target_V)
        critic_loss = layers.reduce_mean(td_error)
        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByNorm(clip_norm=0.5),
            param_list=self.model.get_critic_params())
        opt_c = fluid.optimizer.AdamOptimizer(self.value_lr)
        opt_c.minimize(critic_loss, parameter_list=self.model.get_critic_params())

        # Actor 更新
        i = self.agent_index
        # 新旧策略的 log_prob
        logits = self.model.policy(obs_n[i])
        dist = SoftPDistribution(logits, self.act_space[i])
        logprob = dist.log_prob(act_n[i])
        with fluid.no_grad():
            old_logits = self.old_policy.forward(obs_n[i])
            old_dist = SoftPDistribution(old_logits, self.act_space[i])
            old_logprob = old_dist.log_prob(act_n[i])
        ratio = layers.exp(logprob - old_logprob)
        advantage = (target_V - V)
        advantage.stop_gradient = True
        surrogate1 = ratio * advantage
        surrogate2 = layers.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        actor_loss = -layers.reduce_mean(layers.elementwise_min(surrogate1, surrogate2))
        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByNorm(clip_norm=0.5),
            param_list=self.model.get_actor_params())
        opt_a = fluid.optimizer.AdamOptimizer(self.policy_lr)
        opt_a.minimize(actor_loss, parameter_list=self.model.get_actor_params())

        # 同步旧策略
        self.model.actor_model.sync_weights_to(self.old_policy)

        return critic_loss
