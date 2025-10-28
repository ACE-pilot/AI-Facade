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
from gym import spaces

# 为了最大兼容：算子统一走包装器
F = fluid.layers
__all__ = ['MAPPO']
EPS = 1e-8

# ===== 数学函数兼容封装 =====
def _exp(x):
    try:
        return F.exp(x)
    except Exception:
        try:
            return paddle.exp(x)
        except Exception:
            from paddle import tensor
            return tensor.exp(x)

def _log(x):
    try:
        return F.log(x)
    except Exception:
        try:
            return paddle.log(x)
        except Exception:
            from paddle import tensor
            return tensor.log(x)

def _tanh(x):
    try:
        return F.tanh(x)
    except Exception:
        try:
            return paddle.tanh(x)
        except Exception:
            import paddle.nn.functional as PF
            return PF.tanh(x)

def _clip(x, min=None, max=None):
    # fluid.layers.clip 可能不存在，退回 paddle.clip
    try:
        return F.clip(x, min=min, max=max)
    except Exception:
        try:
            return paddle.clip(x, min=min, max=max)
        except Exception:
            from paddle import tensor
            return tensor.clip(x, min=min, max=max)

# ===== 动作分布 =====
def SoftPDistribution(logits, act_space):
    # 单离散
    if isinstance(act_space, spaces.Discrete) or hasattr(act_space, 'n'):
        return SoftCategoricalDistribution(logits)

    # 多离散
    if isinstance(act_space, spaces.MultiDiscrete) or hasattr(act_space, 'nvec'):
        low = getattr(act_space, 'low', None)
        high = getattr(act_space, 'high', None)
        if low is None or high is None:
            nvec = np.array(act_space.nvec)
            low = np.zeros_like(nvec)
            high = nvec - 1
        return SoftMultiCategoricalDistribution(logits, low, high)

    # 连续
    if isinstance(act_space, spaces.Box):
        if isinstance(logits, (list, tuple)) and len(logits) == 2:
            means, second = logits
        else:
            # 兼容：自动均分
            D = int(F.shape(logits)[-1] // 2)
            means = logits[:, :D]
            second = logits[:, D:]
        bounded = bool(np.isfinite(act_space.low).all() and np.isfinite(act_space.high).all())
        return GaussianDistribution(means, second, bounded)

    raise AssertionError(f"Unsupported action space: {act_space}")

class GaussianDistribution:
    """对角高斯；支持传入 (mu, log_std) 或 (mu, std)；可选 tanh-squash。"""
    def __init__(self, means, std_or_logstd, bounded=False):
        self.means = means
        # 优先当作 log_std；失败则当作 std
        try:
            self.log_std = std_or_logstd
            self.std = _exp(self.log_std)
        except Exception:
            self.std = std_or_logstd
            self.log_std = _log(_clip(self.std, min=EPS))
        self.std = _clip(self.std, min=1e-6, max=1e2)
        self.log_std = _clip(self.log_std, min=-20.0, max=5.0)
        self.bounded = bounded

    def _normal_sample(self):
        noise = paddle.randn(self.means.shape, dtype=self.means.dtype)
        return self.means + noise * self.std

    def sample(self):
        if not self.bounded:
            return self._normal_sample()
        raw = self._normal_sample()
        return _tanh(raw)

    def _log_prob_normal(self, actions):
        var = F.square(self.std) + EPS
        log_scale = 0.5 * _log(2.0 * np.pi * var)
        term = -0.5 * F.square(actions - self.means) / var
        return F.reduce_sum(term - log_scale, dim=-1)

    def log_prob(self, actions):
        if not self.bounded:
            return self._log_prob_normal(actions)
        clipped = _clip(actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        raw = 0.5 * (_log(1.0 + clipped + EPS) - _log(1.0 - clipped + EPS))
        base_lp = self._log_prob_normal(raw)
        log_det_j = F.reduce_sum(_log(1.0 - F.square(clipped) + EPS), dim=-1)
        return base_lp - log_det_j

    def entropy(self):
        try:
            D = self.means.shape[-1]
            D = paddle.to_tensor(D, dtype=self.means.dtype)
        except Exception:
            return F.reduce_sum(self.log_std, dim=-1)
        const = 0.5 * (1.0 + np.log(2 * np.pi))
        return F.reduce_sum(self.log_std, dim=-1) + const * D

# ===== 算法（最小改动成 MAPPO） =====
class MAPPO(Algorithm):
    """
    Multi-Agent PPO with centralized value (V(s)) and decentralized actors.
    —— 保持原接口/依赖/调用方式一致：predict/sample/Q/learn
    """
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=0.99,
                 policy_lr=1e-4,
                 value_lr=1e-3,
                 epsilon=0.2,
                 entropy_coef=0.01,
                 value_clip_coef=0.0,
                 max_grad_norm=0.5):
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        self.model = model
        self.old_policy = deepcopy(model.actor_model)  # 与你原版保持一致
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_clip_coef = value_clip_coef
        self.max_grad_norm = max_grad_norm

        self._actor_params = self.model.get_actor_params()
        self._critic_params = self.model.get_critic_params()

        # 优化器（动态图需在初始化时绑定 parameter_list）
        grad_clip_actor = self._make_grad_clip(self.max_grad_norm)
        grad_clip_critic = self._make_grad_clip(self.max_grad_norm)
        self._opt_actor  = self._make_adam(self.policy_lr, grad_clip_actor,  self._actor_params)
        self._opt_critic = self._make_adam(self.value_lr,  grad_clip_critic, self._critic_params)

    # —— 兼容不同 Paddle 版本 —— #
    def _make_grad_clip(self, clip_norm):
        try:
            return paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm)
        except Exception:
            try:
                from paddle.nn import ClipGradByGlobalNorm
                return ClipGradByGlobalNorm(clip_norm)
            except Exception:
                return None

    def _make_adam(self, lr, grad_clip_obj, param_list):
        try:
            if grad_clip_obj is not None:
                return fluid.optimizer.AdamOptimizer(lr, grad_clip=grad_clip_obj, parameter_list=param_list)
            else:
                return fluid.optimizer.AdamOptimizer(lr, parameter_list=param_list)
        except TypeError:
            try:
                return fluid.optimizer.AdamOptimizer(lr, parameter_list=param_list)
            except Exception:
                return fluid.optimizer.AdamOptimizer(lr)

    # —— 去中心化 actor —— #
    def predict(self, obs):
        logits = self.model.policy(obs)
        dist = SoftPDistribution(logits, self.act_space[self.agent_index])
        return dist.sample()

    # 兼容 simple_agent.sample(..., use_target_model=...) 的多余参数
    def sample(self, obs, **kwargs):
        return self.predict(obs)

    # —— 中心化 critic（接口保留） —— #
    def Q(self, obs_n, act_n):
        return self.model.value(obs_n, act_n)

    def _sync_old_policy(self):
        if hasattr(self.model.actor_model, 'sync_weights_to'):
            self.model.actor_model.sync_weights_to(self.old_policy)
        else:
            src = dict(self.model.actor_model.named_parameters())
            dst = dict(self.old_policy.named_parameters())
            for k in dst:
                if k in src:
                    dst[k].set_value(src[k])

    def _adv_norm(self, adv):
        mean = F.reduce_mean(adv)
        var = F.reduce_mean(F.square(adv - mean))
        std = F.sqrt(var + EPS)
        return (adv - mean) / (std + EPS)

    def learn(self, obs_n, act_n, rewards, next_obs_n, dones):
        """
        最小改动到 MAPPO：
        - 用 centralized V(s)（保持接口 value(obs_n, act_n)，但算法上当 V(s) 用）
        - A = r + γ(1-d)V(s') - V(s)，并做优势标准化
        - Actor 用 PPO-clip + 熵正则
        注：保留单步接口以保证与现有数据流兼容
        """
        # === Critic ===
        V_s   = self.model.value(obs_n, act_n)
        V_nxt = self.model.value(next_obs_n, act_n)

        if len(F.shape(V_s)) > 1 and F.shape(V_s)[-1] == 1:
            V_s = F.squeeze(V_s, axes=[-1])
        if len(F.shape(V_nxt)) > 1 and F.shape(V_nxt)[-1] == 1:
            V_nxt = F.squeeze(V_nxt, axes=[-1])

        rewards = F.squeeze(rewards, axes=[-1]) if len(F.shape(rewards)) > 1 else rewards
        dones   = F.squeeze(dones,   axes=[-1]) if len(F.shape(dones))   > 1 else dones
        rewards = F.cast(rewards, V_s.dtype)
        dones   = F.cast(dones,   V_s.dtype)

        target_V = rewards + self.gamma * (1.0 - dones) * V_nxt

        if self.value_clip_coef > 0.0:
            V_clipped = V_s + _clip(target_V - V_s, min=-self.value_clip_coef, max=self.value_clip_coef)
            v1 = F.square_error_cost(V_s,       target_V)
            v2 = F.square_error_cost(V_clipped, target_V)
            critic_loss = F.reduce_mean(F.elementwise_max(v1, v2))
        else:
            td = F.square_error_cost(V_s, target_V)
            critic_loss = F.reduce_mean(td)

        self._opt_critic.minimize(critic_loss)

        # === Advantage ===
        adv = target_V - V_s
        adv.stop_gradient = True
        adv = self._adv_norm(adv)

        # === Actor (PPO-clip + entropy) ===
        i = self.agent_index
        logits = self.model.policy(obs_n[i])
        dist   = SoftPDistribution(logits, self.act_space[i])
        act_i  = act_n[i]
        logp   = dist.log_prob(act_i)

        old_logits = self.old_policy.forward(obs_n[i])
        old_dist   = SoftPDistribution(old_logits, self.act_space[i])
        old_logp   = old_dist.log_prob(act_i)

        ratio = F.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = _clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

        try:
            ent = F.reduce_mean(dist.entropy())
        except Exception:
            ent = F.reduce_mean(F.zeros_like(logp))

        actor_loss = -F.reduce_mean(F.elementwise_min(surr1, surr2)) - self.entropy_coef * ent
        self._opt_actor.minimize(actor_loss)

        self._sync_old_policy()
        return critic_loss
