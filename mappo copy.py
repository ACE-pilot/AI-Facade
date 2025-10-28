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
        # 数值稳定：强夹紧，防止 log(0) 与 atanh 爆炸
        if not isinstance(actions, paddle.Tensor):
            actions = paddle.to_tensor(actions, dtype=self.means.dtype)

        if not self.bounded:
            return self._log_prob_normal(actions)

        # 1) 强力夹紧动作，留足安全边界
        clipped = paddle.clip(actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)

        # 2) 稳定的 atanh：atanh(x) = 0.5 * (log1p(x) - log1p(-x))
        raw = 0.5 * (paddle.log1p(clipped) - paddle.log1p(-clipped))

        # 3) 基础正态对数概率
        base_lp = self._log_prob_normal(raw)

        # 4) tanh 的雅可比项：sum log(1 - x^2)
        denom = 1.0 - clipped * clipped
        denom = paddle.clip(denom, min=1e-6)        # 防止 log(0)
        log_det_j = paddle.sum(paddle.log(denom), axis=-1)

        return base_lp - log_det_j


    def entropy(self):
        D = int(self.means.shape[-1])
        const = 0.5 * (1.0 + np.log(2 * np.pi))
        return paddle.sum(self.log_std, axis=-1) + const * D

# ===== 算法（MAPPO） =====
class MAPPO(Algorithm):
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
        self.old_policy = deepcopy(model.actor_model)  # 旧策略副本
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = float(gamma)
        self.policy_lr = float(policy_lr)
        self.value_lr  = float(value_lr)
        self.epsilon = float(epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_clip_coef = float(value_clip_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.sync_every = 10    # 每 10 次 learn 再同步 old_policy（可调 5~20）
        self._sync_cnt  = 0


        # —— 收集可训练参数（先尝试 model.get_*，失败则直接从子模块抓）——
        def _as_list(x):
            if x is None:
                return None
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, dict):
                return list(x.values())
            try:
                return list(x)
            except Exception:
                return None

        actor_params = None
        critic_params = None
        # 方式一：模型自带接口
        if hasattr(self.model, "get_actor_params"):
            actor_params = _as_list(self.model.get_actor_params())
        if hasattr(self.model, "get_critic_params"):
            critic_params = _as_list(self.model.get_critic_params())
        # 方式二：直接从子模块抓
        if not actor_params:
            try:
                actor_params = list(self.model.actor_model.parameters())
            except Exception:
                actor_params = None
        if not critic_params:
            try:
                critic_params = list(self.model.critic_model.parameters())
            except Exception:
                critic_params = None

        assert actor_params and critic_params, "MAPPO: failed to collect trainable parameters."

        self._actor_params  = actor_params
        self._critic_params = critic_params

        # —— 梯度裁剪对象（兼容不同 Paddle 版本）——
        self.grad_clip_actor  = self._make_grad_clip(self.max_grad_norm)
        self.grad_clip_critic = self._make_grad_clip(self.max_grad_norm)

        # —— 优化器：动态图优先 paddle.optimizer.Adam，静态图兜底 fluid —— 
        try:
            if paddle.in_dynamic_mode():
                self._opt_actor  = paddle.optimizer.Adam(learning_rate=self.policy_lr, parameters=self._actor_params)
                self._opt_critic = paddle.optimizer.Adam(learning_rate=self.value_lr,  parameters=self._critic_params)
            else:
                raise RuntimeError("fallback to fluid")
        except Exception:
            # 兜底：fluid 优化器
            if self.grad_clip_actor is not None:
                self._opt_actor  = fluid.optimizer.AdamOptimizer(self.policy_lr, grad_clip=self.grad_clip_actor,
                                                                 parameter_list=self._actor_params)
            else:
                self._opt_actor  = fluid.optimizer.AdamOptimizer(self.policy_lr, parameter_list=self._actor_params)
            if self.grad_clip_critic is not None:
                self._opt_critic = fluid.optimizer.AdamOptimizer(self.value_lr,  grad_clip=self.grad_clip_critic,
                                                                 parameter_list=self._critic_params)
            else:
                self._opt_critic = fluid.optimizer.AdamOptimizer(self.value_lr,  parameter_list=self._critic_params)


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
            if paddle.in_dynamic_mode():
                return paddle.optimizer.Adam(learning_rate=lr, parameters=param_list)
        except Exception:
            pass
        try:
            if grad_clip_obj is not None:
                return fluid.optimizer.AdamOptimizer(lr, grad_clip=grad_clip_obj, parameter_list=param_list)
            else:
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
        # ===== Critic 前向 =====
        V_s   = self.model.value(obs_n, act_n)
        V_nxt = self.model.value(next_obs_n, act_n)
        if isinstance(V_s, paddle.Tensor) and V_s.ndim > 1 and int(V_s.shape[-1]) == 1:
            V_s = paddle.squeeze(V_s, axis=[-1])
        if isinstance(V_nxt, paddle.Tensor) and V_nxt.ndim > 1 and int(V_nxt.shape[-1]) == 1:
            V_nxt = paddle.squeeze(V_nxt, axis=[-1])

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

        # —— 数值健康检查（critic）——
        if not paddle.isfinite(critic_loss):
            # 跳过这个 batch，避免把 NaN 反向进网络
            return paddle.to_tensor(0.0, dtype='float32')


        # ==== 调试：只前 3 次记录一次参数均值，用于对比 ====
        debug_critic = not hasattr(self, "_dbg_c") or self._dbg_c < 3
        if debug_critic:
            try:
                cname, cparam = next(iter(self._critic_params)) if isinstance(self._critic_params, dict) \
                    else (None, next(iter(self._critic_params)))
            except Exception:
                cname, cparam = None, None
            c_before = float(paddle.mean(cparam)) if cparam is not None else None

        # ===== Critic 反传/更新（动态图优先）=====
        if paddle.in_dynamic_mode():
            self._opt_critic.clear_grad()
            critic_loss.backward()
            try:
                paddle.nn.utils.clip_grad_norm_(self._critic_params, max_norm=self.max_grad_norm)
            except Exception:
                pass
            self._opt_critic.step()
        else:
            self._opt_critic.minimize(critic_loss)

        if debug_critic:
            try:
                cname2, cparam2 = next(iter(self._critic_params)) if isinstance(self._critic_params, dict) \
                    else (None, next(iter(self._critic_params)))
                c_after = float(paddle.mean(cparam2)) if cparam2 is not None else None
                print(f"[DBG/MAPPO] critic param mean: {c_before} -> {c_after}")
            except Exception:
                print("[DBG/MAPPO] critic param mean: (unavailable)")
            self._dbg_c = getattr(self, "_dbg_c", 0) + 1

        # ===== Advantage（停止梯度并标准化）=====
        adv = target_V - V_s
        adv.stop_gradient = True
        adv = self._adv_norm(adv)

        # ===== Actor 前向 =====
        i = self.agent_index
        logits = self.model.policy(obs_n[i])
        dist   = SoftPDistribution(logits, self.act_space[i])
        act_i  = self._norm_action_for_logprob(act_n[i], self.act_space[i])
        logp   = dist.log_prob(act_i)

        old_logits = self.old_policy.forward(obs_n[i])
        old_dist   = SoftPDistribution(old_logits, self.act_space[i])
        old_logp   = old_dist.log_prob(act_i)

        # 防止 exp 溢出：先裁剪 log_ratio 再做 exp
        log_ratio = logp - old_logp
        log_ratio = paddle.clip(log_ratio, min=-20.0, max=20.0)
        ratio = paddle.exp(log_ratio)

        surr1 = ratio * adv
        surr2 = _clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv

        try:
            ent = paddle.mean(dist.entropy())
        except Exception:
            ent = paddle.mean(paddle.zeros_like(logp))

        actor_loss = -paddle.mean(paddle.minimum(surr1, surr2)) - self.entropy_coef * ent

        # —— 数值健康检查（actor）——
        safe_tensors = [logp, old_logp, ratio, actor_loss]
        for t in safe_tensors:
            if not paddle.isfinite(t).all():
                return paddle.to_tensor(0.0, dtype='float32')


        # ==== 调试：只前 3 次记录一次 actor 参数均值，用于对比 ====
        debug_actor = not hasattr(self, "_dbg_a") or self._dbg_a < 3
        if debug_actor:
            try:
                aname, aparam = next(iter(self._actor_params)) if isinstance(self._actor_params, dict) \
                    else (None, next(iter(self._actor_params)))
            except Exception:
                aname, aparam = None, None
            a_before = float(paddle.mean(aparam)) if aparam is not None else None

        # ===== Actor 反传/更新（动态图优先）=====
        if paddle.in_dynamic_mode():
            self._opt_actor.clear_grad()
            actor_loss.backward()
            try:
                paddle.nn.utils.clip_grad_norm_(self._actor_params, max_norm=self.max_grad_norm)
            except Exception:
                pass
            self._opt_actor.step()
        else:
            self._opt_actor.minimize(actor_loss)

        if debug_actor:
            try:
                aname2, aparam2 = next(iter(self._actor_params)) if isinstance(self._actor_params, dict) \
                    else (None, next(iter(self._actor_params)))
                a_after = float(paddle.mean(aparam2)) if aparam2 is not None else None
                print(f"[DBG/MAPPO] actor  param mean: {a_before} -> {a_after}")
            except Exception:
                print("[DBG/MAPPO] actor  param mean: (unavailable)")
            self._dbg_a = getattr(self, "_dbg_a", 0) + 1

        # 固定 old_policy：每次更新后同步一次
        self._sync_cnt += 1
        if self._sync_cnt % self.sync_every == 0:
            self._sync_old_policy()

        return critic_loss
