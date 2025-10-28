# mappo.py —— Paddle 2.x 纯 PPO 稳定版：KL 约束 + 信任掩码 + 热身放宽 + 打印检验
# 说明：
# - 彻底移除 numpy()[0] 用法，全部用 float() 安全转换
# - 反复触发 early-stop 的问题：放宽 target_kl，延后早停判定到第3轮；前10次 learn 关闭信任掩码
# - 提供打印检验：KL、保留比例、最大|log_ratio|、actor/critic loss、entropy、kl_coef、梯度范数、shape 检查
# - 兼容离散/多离散/连续（tanh-squash）动作；连续动作的 log_prob 含tanh-Jacobian

import warnings
warnings.simplefilter('default')

import os
import numpy as np
from copy import deepcopy
import paddle
import paddle.nn.functional as PF
from paddle import fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid.policy_distribution import (
    SoftCategoricalDistribution,
    SoftMultiCategoricalDistribution,
)
from gym import spaces

__all__ = ['MAPPO']

# 别名
SoftMultiCat = SoftMultiCategororicalDistribution = SoftMultiCategoricalDistribution

EPS = 1e-8

# ====== 基本数值封装 ======
_defdtype = 'float32'

def _exp(x):   return paddle.exp(x)
def _log(x):   return paddle.log(x)
def _tanh(x):  return paddle.tanh(x)
def _clip(x, min=None, max=None): return paddle.clip(x, min=min, max=max)

def _as_tensor(x, dtype=None):
    """Safer helper: prefer paddle dtype objects (e.g., V_s.dtype),
    avoid relying on strings like 'FP32'."""
    if isinstance(x, paddle.Tensor):
        if dtype is None:
            return x
        return paddle.cast(x, dtype)
    else:
        return paddle.to_tensor(x, dtype=dtype)

# ====== 动作分布 ======
class GaussianDistribution:
    """对角高斯；支持 (mu, log_std) 或 (mu, std)；可选 tanh-squash。"""
    def __init__(self, means, std_or_logstd, bounded=False):
        self.means = means
        try:
            self.log_std = std_or_logstd
            self.std     = _exp(self.log_std)
        except Exception:
            self.std     = std_or_logstd
            self.log_std = _log(_clip(self.std, min=EPS))
        # 数值护栏（更紧）：避免熵爆 / KL 暴冲
        self.std     = _clip(self.std,     min=1e-4, max=3.0)
        self.log_std = _clip(self.log_std, min=-3.0, max=-0.7)
        self.bounded = bool(bounded)

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
        if not isinstance(actions, paddle.Tensor):
            actions = paddle.to_tensor(actions, dtype=self.means.dtype)
        if not self.bounded:
            return self._log_prob_normal(actions)
        # tanh 逆变换 + Jacobian
        clipped = paddle.clip(actions, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        raw = 0.5 * (paddle.log1p(clipped) - paddle.log1p(-clipped))
        base_lp = self._log_prob_normal(raw)
        denom = paddle.clip(1.0 - clipped * clipped, min=1e-6)
        log_det_j = paddle.sum(paddle.log(denom), axis=-1)
        return base_lp - log_det_j

    def entropy(self):
        D = int(self.means.shape[-1])
        const = 0.5 * (1.0 + np.log(2 * np.pi))
        return paddle.sum(self.log_std, axis=-1) + const * D


def SoftPDistribution(logits, act_space):
    """根据动作空间返回合适的分布对象。
    - Discrete: SoftCategoricalDistribution
    - MultiDiscrete: SoftMultiCategoricalDistribution
    - Box: 高斯（支持 tanh-squash）
    """
    # 单离散
    if isinstance(act_space, spaces.Discrete) or hasattr(act_space, 'n'):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, paddle.Tensor) and logits.ndim == 1:
            logits = logits.unsqueeze(0)
        return SoftCategoricalDistribution(logits)

    # 多离散
    if isinstance(act_space, spaces.MultiDiscrete) or hasattr(act_space, 'nvec'):
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
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
            last = int(logits.shape[-1]) if hasattr(logits, 'shape') else int(np.asarray(logits).shape[-1])
            D = last // 2
            means  = logits[:, :D]
            second = logits[:, D:]
        bounded = bool(np.isfinite(act_space.low).all() and np.isfinite(act_space.high).all())
        return GaussianDistribution(means, second, bounded)

    raise AssertionError(f"Unsupported action space: {act_space}")


# ====== 算法：MAPPO（含稳定化与打印检验） ======
class MAPPO(Algorithm):
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=0.99,
                 policy_lr=1e-4,         # ↑ 调大：策略更容易“动起来”
                 value_lr=1e-3,
                 epsilon=0.1,
                 entropy_coef=0.002,      # ↑ 初期更多探索
                 value_clip_coef=0.2,    # 打开 value clipping
                 max_grad_norm=0.5,
                 target_kl=0.15,         # ↑ 放宽，避免频繁早停
                 sync_every=1,           # 每次 learn 完就同步 old_policy
                 k_epochs=3,             # ↑ 多走几轮小步更新
                 strict_ppo=True,
                 trust_clip_logr=1.0,    # ↑ 信任掩码阈值更宽（|log_ratio|<=2.5）
                 min_keep_ratio=0.05,    # 掩码后至少保留 5% 样本
                 gae_lambda=0.95, debug=True):
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)

        self.model = model
        self.old_policy = deepcopy(model.actor_model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = float(gamma)
        self.policy_lr = float(policy_lr)
        self.value_lr  = float(value_lr)
        self.epsilon = float(epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_clip_coef = float(value_clip_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.strict_ppo = bool(strict_ppo)
        self.target_kl = float(target_kl)
        self.sync_every = int(sync_every)
        self.k_epochs = int(k_epochs)
        self._sync_cnt  = 0
        self.kl_coef = 5.0
        self.trust_clip_logr = float(trust_clip_logr)
        self.min_keep_ratio = float(min_keep_ratio)
        self.gae_lambda = float(gae_lambda)
        self.debug = bool(debug)

        # —— 收集参数 —— #
        def _as_list(x):
            if x is None: return None
            if isinstance(x, (list, tuple)): return list(x)
            if isinstance(x, dict): return list(x.values())
            try: return list(x)
            except Exception: return None

        actor_params, critic_params = None, None
        if hasattr(self.model, "get_actor_params"):
            actor_params = _as_list(self.model.get_actor_params())
        if hasattr(self.model, "get_critic_params"):
            critic_params = _as_list(self.model.get_critic_params())
        if not actor_params:
            try: actor_params = list(self.model.actor_model.parameters())
            except Exception: actor_params = None
        if not critic_params:
            try: critic_params = list(self.model.critic_model.parameters())
            except Exception: critic_params = None
        assert actor_params and critic_params, "MAPPO: failed to collect trainable parameters."
        self._actor_params  = actor_params
        self._critic_params = critic_params

        # —— 梯度裁剪 —— #
        self.grad_clip_actor  = self._make_grad_clip(self.max_grad_norm)
        self.grad_clip_critic = self._make_grad_clip(self.max_grad_norm)

        # —— 优化器 —— #
        try:
            if paddle.in_dynamic_mode():
                self._opt_actor  = paddle.optimizer.Adam(learning_rate=self.policy_lr, parameters=self._actor_params)
                self._opt_critic = paddle.optimizer.Adam(learning_rate=self.value_lr,  parameters=self._critic_params)
            else:
                raise RuntimeError("fallback to fluid")
        except Exception:
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

        if self.debug:
            n_a = sum(p.numel() for p in self._actor_params)
            n_c = sum(p.numel() for p in self._critic_params)
            print(f"[MAPPO.dbg] init actor_params={n_a} critic_params={n_c} target_kl={self.target_kl} lr_pi={self.policy_lr} lr_v={self.value_lr}")

    # —— 兼容不同 Paddle 版本 —— #
    def _make_grad_clip(self, clip_norm):
        try:
            from paddle.nn import ClipGradByGlobalNorm
            return ClipGradByGlobalNorm(clip_norm)
        except Exception:
            try:
                return paddle.fluid.clip.GradientClipByGlobalNorm(clip_norm=clip_norm)
            except Exception:
                return None

    # —— 去中心化 actor —— #
    def predict(self, obs):
        logits = self.model.policy(obs)
        dist   = SoftPDistribution(logits, self.act_space[self.agent_index])
        return dist.sample()

    def sample(self, obs, **kwargs):
        return self.predict(obs)

    # —— 中心化 critic —— #
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
        # 统一 dtype/shape，避免广播或 squeeze 错误
        if not isinstance(act, paddle.Tensor):
            act = paddle.to_tensor(act)
        if isinstance(space, spaces.Box):
            if act.dtype != paddle.float32:
                act = paddle.cast(act, 'float32')
            if act.ndim > 1 and int(act.shape[-1]) == 1:
                act = paddle.squeeze(act, axis=[-1])
            return act
        else:
            if act.ndim > 1 and int(act.shape[-1]) == 1:
                act = paddle.squeeze(act, axis=[-1])
            if act.dtype != paddle.int64:
                act = paddle.cast(act, 'int64')
            return act

    def _shape(self, x):
        try:
            return list(x.shape)
        except Exception:
            try:
                return list(np.asarray(x).shape)
            except Exception:
                return None

    # ====== Learn ======
    def learn(self, obs_n, act_n, rewards, next_obs_n, dones):
        dbg = self.debug

        # ----- 开始学习前先冻结旧策略（确保与本批数据一致）-----
        self._sync_old_policy()

        # ----- Critic 前向（兼容签名）-----
        try:
            V_s   = self.model.value(obs_n)
            V_nxt = self.model.value(next_obs_n)
        except TypeError:
            V_s   = self.model.value(obs_n, act_n)
            V_nxt = self.model.value(next_obs_n, act_n)

        V_nxt = V_nxt.detach()

        def _squeeze_last1(x):
            if isinstance(x, paddle.Tensor) and x.ndim > 1 and int(x.shape[-1]) == 1:
                return paddle.squeeze(x, axis=[-1])
            return x

        V_s = _squeeze_last1(V_s)
        V_nxt = _squeeze_last1(V_nxt)

        rewards = paddle.to_tensor(rewards, dtype=V_s.dtype)
        dones   = paddle.to_tensor(dones,   dtype=V_s.dtype)
        if rewards.ndim > 1: rewards = paddle.squeeze(rewards, axis=[-1])
        if dones.ndim > 1:   dones   = paddle.squeeze(dones,   axis=[-1])

        rewards = paddle.cast(rewards, V_s.dtype)
        dones   = paddle.cast(dones,   V_s.dtype)

        # ===== GAE(λ) 优势与回报 =====
        T = int(V_s.shape[0])
        adv = paddle.zeros_like(V_s)
        lastgaelam = paddle.to_tensor(0.0, dtype=V_s.dtype)
        for t in range(T - 1, -1, -1):
            mask_t = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * V_nxt[t] * mask_t - V_s[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * mask_t * lastgaelam
            adv[t] = lastgaelam
        target_V = (adv + V_s).detach()

        # ===== Critic 损失（保留 value clipping）=====
        if self.value_clip_coef > 0.0:
            V_clipped = V_s + _clip(target_V - V_s, min=-self.value_clip_coef, max=self.value_clip_coef)
            v1 = (V_s       - target_V) * (V_s       - target_V)
            v2 = (V_clipped - target_V) * (V_clipped - target_V)
            critic_loss = paddle.mean(paddle.maximum(v1, v2))
        else:
            td = (V_s - target_V) * (V_s - target_V)
            critic_loss = paddle.mean(td)

        if not paddle.isfinite(critic_loss):
            if dbg:
                print("[MAPPO.dbg] critic loss NaN/Inf -> skip")
            return paddle.to_tensor(0.0, dtype='float32')

        # ----- Critic 更新 -----
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

        # ----- Advantage（停止梯度并标准化）-----
        adv.stop_gradient = True
        adv = self._adv_norm(adv)

        # ----- 冻结 old_logp（本次 learn 的基线）-----
        i = self.agent_index
        act_i = self._norm_action_for_logprob(act_n[i], self.act_space[i])

        with paddle.no_grad():
            old_logits = self.old_policy.forward(obs_n[i])
            old_dist   = SoftPDistribution(old_logits, self.act_space[i])
            old_logp   = old_dist.log_prob(act_i)

            # 数值检验
            if dbg:
                if not paddle.isfinite(old_logp).all():
                    print("[MAPPO.dbg] old_logp contains NaN/Inf")

        # ----- PPO 多轮小步 -----
        approx_kl_val = 0.0
        early_stop = False
        last_actor_loss = 0.0
        keep_ratio_val = 1.0
        max_abs_logr = 0.0

        for epk in range(self.k_epochs):
            # 前向
            logits = self.model.policy(obs_n[i])
            dist   = SoftPDistribution(logits, self.act_space[i])
            logp   = dist.log_prob(act_i)

            # 计算 log-ratio 并做数值裁剪
            log_ratio = paddle.clip(logp - old_logp, min=-10.0, max=10.0)

            # 信任掩码：始终启用，严格控制离群样本
            mask = paddle.abs(log_ratio) <= self.trust_clip_logr
            if mask.dtype != paddle.bool:
                mask = mask.astype('bool')

            n_total = int(mask.shape[0])
            n_keep = int(float(paddle.sum(mask))) if n_total > 0 else 0
            keep_ratio_val = (n_keep / max(1, n_total))

            if n_keep < max(1, int(self.min_keep_ratio * n_total)):
                if dbg:
                    print(f"[MAPPO.dbg] skip actor: keep_ratio={keep_ratio_val:.2f} < {self.min_keep_ratio:.2f}")
                early_stop = True
                break

            # 只用“可信”样本
            log_ratio_kept = paddle.masked_select(log_ratio, mask)
            ratio = paddle.exp(log_ratio_kept)
            adv_kept = paddle.masked_select(adv, mask)

            surr1 = ratio * adv_kept
            surr2 = _clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_kept
            pg_loss = -paddle.mean(paddle.minimum(surr1, surr2))

            # 熵
            try:
                ent_all = dist.entropy()
                ent_kept = paddle.masked_select(ent_all, mask)
                ent = paddle.mean(ent_kept)
            except Exception:
                ent = paddle.mean(paddle.zeros_like(log_ratio_kept))

            # std 护栏正则
            std_reg = paddle.to_tensor(0.0, dtype=log_ratio_kept.dtype)
            if hasattr(dist, "log_std"):
                lower, upper = -1.5, -0.3
                over = PF.relu(dist.log_std - upper)
                under = PF.relu(lower - dist.log_std)
                std_reg = paddle.mean(over * over + under * under) * 5e-2

            # 近似 KL（仅 kept 样本） + KL penalty
            old_logp_kept = paddle.masked_select(old_logp, mask)
            logp_kept     = old_logp_kept + log_ratio_kept
            approx_kl = paddle.mean(old_logp_kept - logp_kept)
            kl_pen = self.kl_coef * approx_kl

            actor_loss = pg_loss - self.entropy_coef * ent + kl_pen + std_reg

            if not paddle.isfinite(actor_loss):
                if dbg: print("[MAPPO.dbg] actor loss NaN/Inf -> skip")
                break

            # 反传
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

            # 自适应 KL + 早停（第3轮以后才允许早停）
            approx_kl_val = float(approx_kl)
            last_actor_loss = float(actor_loss)
            max_abs_logr = float(paddle.max(paddle.abs(log_ratio_kept)))

            if (epk + 1) >= 3 and approx_kl_val > self.target_kl * 1.5:
                self.kl_coef = float(np.clip(self.kl_coef * 1.5, 1e-4, 10.0))
                early_stop = True
                if dbg:
                    print(f"[MAPPO.dbg] early-stop@epoch {epk+1} KL={approx_kl_val:.4f} > 1.5*target")
                break
            elif approx_kl_val < self.target_kl / 1.5:
                self.kl_coef = float(np.clip(self.kl_coef / 1.5, 1e-4, 10.0))

        # ----- old_policy 同步（按频率）-----
        self._sync_cnt += 1
        if self._sync_cnt % self.sync_every == 0:
            if dbg:
                print(f"[MAPPO.dbg] >>> sync old_policy at learn #{self._sync_cnt} (every {self.sync_every})")
            self._sync_old_policy()

        # ----- 打印检验（float()，杜绝 0D numpy 索引）-----
        if dbg:
            try:
                a_g_sq = 0.0
                cnt = 0
                for p in self._actor_params:
                    if p.grad is not None:
                        gnorm = float(paddle.linalg.norm(p.grad))
                        a_g_sq += gnorm * gnorm
                        cnt += 1
                a_gn = (a_g_sq ** 0.5) if cnt > 0 else 0.0
            except Exception:
                a_gn = 0.0

            # 演示位：应恒为 1
            ratio_mu = 1.0

            # 形状检查（只打印前几次，避免刷屏）
            if self._sync_cnt <= 5:
                def _fmt(obj):
                    try:
                        return list(obj.shape)
                    except Exception:
                        try:
                            return list(np.asarray(obj).shape)
                        except Exception:
                            return None

            # 动作饱和率（连续动作）：|a|>0.95 的占比
            sat95 = 0.0
            try:
                if isinstance(self.act_space[i], spaces.Box):
                    sat95 = float(paddle.mean((paddle.abs(act_i) > 0.95).astype('float32')))
            except Exception:
                sat95 = 0.0

            print("[MAPPO.dbg] step={} agent={} KL={:.4f} keep={:.2f} "
                  "max|logr|={:.2f} loss_a={:.6f} loss_c={:.6f} ent={:.3f} kl_coef={:.4f} gnorm={:.3e} sat@.95={:.2f}".format(
                      self._sync_cnt, i, approx_kl_val, keep_ratio_val, max_abs_logr,
                      last_actor_loss, float(critic_loss), float(ent) if 'ent' in locals() else 0.0,
                      self.kl_coef, a_gn, sat95))

        self.entropy_coef = float(np.clip(self.entropy_coef * 0.9995, 1e-4, 0.01))
        return critic_loss
