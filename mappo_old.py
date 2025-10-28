# mappo.py  —— Paddle 2.x 兼容版（去 fluid.layers 运算依赖；动作/形状自适应；接口不变）
import warnings
warnings.simplefilter('default')

import numpy as np
from copy import deepcopy
import paddle
import paddle.nn.functional as PF  # 预留：目前未直接使用
from paddle import fluid  # 仅为优化器最大兼容；其余运算全部走 paddle.*
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid.policy_distribution import (
    SoftCategoricalDistribution,
    SoftMultiCategoricalDistribution,
)
# 统一别名，后文用 SoftMultiCat
SoftMultiCat = SoftMultiCategoricalDistribution

from gym import spaces


__all__ = ['MAPPO']
EPS = 1e-8

# ===== 基本数学封装（直连 paddle.*，不再使用 fluid.layers） =====
def _exp(x):   return paddle.exp(x)
def _log(x):   return paddle.log(x)
def _tanh(x):  return paddle.tanh(x)
def _clip(x, min=None, max=None): return paddle.clip(x, min=min, max=max)

# ===== 动作分布封装 =====
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
        return SoftMultiCat(logits, low, high)

    # 连续
    if isinstance(act_space, spaces.Box):
        if isinstance(logits, (list, tuple)) and len(logits) == 2:
            means, second = logits
        else:
            last = int(logits.shape[-1]) if hasattr(logits, 'shape') else int(np.asarray(logits).shape[-1])
            D = last // 2
            means  = logits[:, :D]
            second = logits[:, D:]
        bounded = bool(np.isfinite(act_space.low).all() and np.isfinite(act_space.high).all())
        return GaussianDistribution(means, second, bounded)

    raise AssertionError(f"Unsupported action space: {act_space}")

class GaussianDistribution:
    """对角高斯；支持 (mu, log_std) 或 (mu, std)；可选 tanh-squash。"""
    def __init__(self, means, std_or_logstd, bounded=False):
        self.means = means
        # 优先当作 log_std；失败则当作 std
        try:
            self.log_std = std_or_logstd
            self.std     = _exp(self.log_std)
        except Exception:
            self.std     = std_or_logstd
            self.log_std = _log(_clip(self.std, min=EPS))
        self.std     = _clip(self.std,     min=1e-6, max=1e2)
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
        var       = paddle.square(self.std) + EPS
        log_scale = 0.5 * _log(2.0 * np.pi * var)
        term      = -0.5 * paddle.square(actions - self.means) / var
        return paddle.sum(term - log_scale, axis=-1)

    def log_prob(self, actions):
        if not self.bounded:
            return self._log_prob_normal(actions)
        clipped   = _clip(actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        raw       = 0.5 * (_log(1.0 + clipped + EPS) - _log(1.0 - clipped + EPS))
        base_lp   = self._log_prob_normal(raw)
        log_det_j = paddle.sum(_log(1.0 - paddle.square(clipped) + EPS), axis=-1)
        return base_lp - log_det_j

    def entropy(self):
        D = int(self.means.shape[-1])
        const = 0.5 * (1.0 + np.log(2 * np.pi))
        return paddle.sum(self.log_std, axis=-1) + const * D

# ===== 算法（MAPPO） =====
class MAPPO(Algorithm):
    """
    Multi-Agent PPO with centralized value (V(s)) and decentralized actors.
    接口保持：predict / sample / Q / learn
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
        self.old_policy = deepcopy(model.actor_model)  # 与原版一致
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = float(gamma)
        self.policy_lr = float(policy_lr)
        self.value_lr  = float(value_lr)
        self.epsilon = float(epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_clip_coef = float(value_clip_coef)
        self.max_grad_norm = float(max_grad_norm)

        self._actor_params  = self.model.get_actor_params()
        self._critic_params = self.model.get_critic_params()

        # 优化器（动态图需初始化时绑定 parameter_list）
        grad_clip_actor  = self._make_grad_clip(self.max_grad_norm)
        grad_clip_critic = self._make_grad_clip(self.max_grad_norm)
        self._opt_actor  = self._make_adam(self.policy_lr, grad_clip_actor,  self._actor_params)
        self._opt_critic = self._make_adam(self.value_lr,  grad_clip_critic, self._critic_params)

    # —— 兼容不同 Paddle 版本（优先 paddle.nn，再退回 fluid） —— #
    def _make_grad_clip(self, clip_norm):
        try:
            from paddle.nn import ClipGradByGlobalNorm
            return ClipGradByGlobalNorm(clip_norm)
        except Exception:
            try:
                return paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm)
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
        dist   = SoftPDistribution(logits, self.act_space[self.agent_index])
        return dist.sample()

    # 兼容 simple_agent.sample(..., use_target_model=...) 的多余参数
    def sample(self, obs, **kwargs):
        return self.predict(obs)

    # —— 中心化 critic（接受并忽略多余 kwargs） —— #
    def Q(self, obs_n, act_n, **kwargs):
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
        mean = paddle.mean(adv)
        var  = paddle.mean((adv - mean) * (adv - mean))
        std  = paddle.sqrt(var + EPS)
        return (adv - mean) / (std + EPS)

    def _norm_action_for_logprob(self, act, space):
        """按动作空间类型规范化动作 dtype/形状，确保 dist.log_prob 可用。"""
        if not isinstance(act, paddle.Tensor):
            act = paddle.to_tensor(act)
        if isinstance(space, spaces.Box):
            # 连续动作：float32，形状 [B, A]
            if act.dtype != paddle.float32:
                act = paddle.cast(act, 'float32')
            return act
        else:
            # 离散 / 多离散：int64，形状 [B] 或 [B, K]
            if act.ndim > 1 and int(act.shape[-1]) == 1:
                act = paddle.squeeze(act, axis=[-1])
            if act.dtype != paddle.int64:
                act = paddle.cast(act, 'int64')
            return act

    def learn(self, obs_n, act_n, rewards, next_obs_n, dones):
        """
        - V(s) 做 centralized critic
        - A = r + γ(1-d)V(s') - V(s) 并做标准化
        - Actor: PPO-clip + entropy
        """
        # === Critic ===
        V_s   = self.model.value(obs_n, act_n)
        V_nxt = self.model.value(next_obs_n, act_n)

        if isinstance(V_s, paddle.Tensor) and V_s.ndim > 1 and int(V_s.shape[-1]) == 1:
            V_s = paddle.squeeze(V_s, axis=[-1])
        if isinstance(V_nxt, paddle.Tensor) and V_nxt.ndim > 1 and int(V_nxt.shape[-1]) == 1:
            V_nxt = paddle.squeeze(V_nxt, axis=[-1])

        # 奖励/终止 squeeze + 类型对齐
        if not isinstance(rewards, paddle.Tensor):
            rewards = paddle.to_tensor(rewards, dtype=V_s.dtype)
        if not isinstance(dones, paddle.Tensor):
            dones = paddle.to_tensor(dones, dtype=V_s.dtype)
        if rewards.ndim > 1:
            rewards = paddle.squeeze(rewards, axis=[-1])
        if dones.ndim > 1:
            dones = paddle.squeeze(dones, axis=[-1])
        rewards = paddle.cast(rewards, V_s.dtype)
        dones   = paddle.cast(dones,   V_s.dtype)

        target_V = rewards + self.gamma * (1.0 - dones) * V_nxt

        if self.value_clip_coef > 0.0:
            V_clipped = V_s + _clip(target_V - V_s, min=-self.value_clip_coef, max=self.value_clip_coef)
            v1 = (V_s       - target_V) * (V_s       - target_V)
            v2 = (V_clipped - target_V) * (V_clipped - target_V)
            critic_loss = paddle.mean(paddle.maximum(v1, v2))
        else:
            td = (V_s - target_V) * (V_s - target_V)
            critic_loss = paddle.mean(td)

        self._opt_critic.minimize(critic_loss)

        # === Advantage ===
        adv = target_V - V_s
        adv.stop_gradient = True
        adv = self._adv_norm(adv)

        # === Actor (PPO-clip + entropy) ===
        i = self.agent_index
        logits = self.model.policy(obs_n[i])
        dist   = SoftPDistribution(logits, self.act_space[i])
        act_i  = self._norm_action_for_logprob(act_n[i], self.act_space[i])
        logp   = dist.log_prob(act_i)

        old_logits = self.old_policy.forward(obs_n[i])
        old_dist   = SoftPDistribution(old_logits, self.act_space[i])
        old_logp   = old_dist.log_prob(act_i)

        ratio = paddle.exp(logp - old_logp)
        surr1 = ratio * adv
        surr2 = _clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

        try:
            ent = paddle.mean(dist.entropy())
        except Exception:
            ent = paddle.mean(paddle.zeros_like(logp))

        actor_loss = -paddle.mean(paddle.minimum(surr1, surr2)) - self.entropy_coef * ent
        self._opt_actor.minimize(actor_loss)

        self._sync_old_policy()
        return critic_loss
