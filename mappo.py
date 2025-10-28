# mappo.py —— Paddle 2.x 纯 PPO 稳定版：KL 约束 + 信任掩码 + GAE(λ) + Mini-batch 切分训练 + 打印检验
# 说明：
# - on-policy，无经验回放；支持按 rollout=T（如 2048）→ mini-batch（如 256）切分；同一批数据上做 k_epochs 次小步更新
# - GAE(λ) 降低优势方差；KL penalty + target_kl 自适应；信任掩码基于 |log_ratio| 抗离群样本
# - 连续动作支持 tanh-squash 与 Jacobian；彻底杜绝 numpy()[0]，用 float() 安全转换

import warnings
warnings.simplefilter('default')

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

SoftMultiCat = SoftMultiCategororicalDistribution = SoftMultiCategoricalDistribution

EPS = 1e-8

# ====== 基本数值封装 ======

def _exp(x):   return paddle.exp(x)
def _log(x):   return paddle.log(x)
def _tanh(x):  return paddle.tanh(x)
def _clip(x, min=None, max=None): return paddle.clip(x, min=min, max=max)

def _as_tensor(x, dtype=None):
    """Safer helper: prefer paddle dtype objects (e.g., V_s.dtype)."""
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
        self.log_std = _clip(self.log_std, min=-3.0, max=-0.7)  # σ ≤ ~0.5
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
    """根据动作空间返回合适的分布对象。"""
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


# ====== 算法：MAPPO（含稳定化 + Mini-batch） ======
class MAPPO(Algorithm):
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=0.99,
                 policy_lr=1e-4,
                 value_lr=1e-3,
                 epsilon=0.1,
                 entropy_coef=0.002,
                 value_clip_coef=0.2,
                 max_grad_norm=0.5,
                 target_kl=0.15,
                 sync_every=1,
                 k_epochs=3,
                 minibatch_size=256,     # ← mini-batch 大小（建议 T=2048, mb=256, k=3~10）
                 strict_ppo=True,
                 trust_clip_logr=1.0,
                 min_keep_ratio=0.05,
                 gae_lambda=0.95,
                 debug=True):
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
        self.minibatch_size = int(minibatch_size)
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

        # —— 优化器 —— #
        try:
            if paddle.in_dynamic_mode():
                self._opt_actor  = paddle.optimizer.Adam(learning_rate=self.policy_lr, parameters=self._actor_params)
                self._opt_critic = paddle.optimizer.Adam(learning_rate=self.value_lr,  parameters=self._critic_params)
            else:
                raise RuntimeError("fallback to fluid")
        except Exception:
            self._opt_actor  = fluid.optimizer.AdamOptimizer(self.policy_lr,  parameter_list=self._actor_params)
            self._opt_critic = fluid.optimizer.AdamOptimizer(self.value_lr,   parameter_list=self._critic_params)

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

    def _squeeze_last1(self, x):
        if isinstance(x, paddle.Tensor) and x.ndim > 1 and int(x.shape[-1]) == 1:
            return paddle.squeeze(x, axis=[-1])
        return x

    def _take(self, x, idx):
        """安全地按 idx 取第一维子集，兼容 numpy/paddle。"""
        if isinstance(x, paddle.Tensor):
            return paddle.index_select(x, paddle.to_tensor(idx, dtype='int64'), axis=0)
        else:
            return x[idx]

    # ====== Learn（含 mini-batch） ======
    def learn(self, obs_n, act_n, rewards, next_obs_n, dones):
        dbg = self.debug

        # —— 冻结旧策略（确保与本批数据一致）—— #
        self._sync_old_policy()

        # ----- Critic 前向 -----
        try:
            V_s   = self.model.value(obs_n)
            V_nxt = self.model.value(next_obs_n)
        except TypeError:
            V_s   = self.model.value(obs_n, act_n)
            V_nxt = self.model.value(next_obs_n, act_n)
        V_nxt = V_nxt.detach()
        V_s = self._squeeze_last1(V_s)
        V_nxt = self._squeeze_last1(V_nxt)

        rewards = paddle.to_tensor(rewards, dtype=V_s.dtype)
        dones   = paddle.to_tensor(dones,   dtype=V_s.dtype)
        if rewards.ndim > 1: rewards = paddle.squeeze(rewards, axis=[-1])
        if dones.ndim > 1:   dones   = paddle.squeeze(dones,   axis=[-1])

        # ===== GAE(λ) =====
        T = int(V_s.shape[0])
        adv = paddle.zeros_like(V_s)
        lastgaelam = paddle.to_tensor(0.0, dtype=V_s.dtype)
        for t in range(T - 1, -1, -1):
            mask_t = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * V_nxt[t] * mask_t - V_s[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * mask_t * lastgaelam
            adv[t] = lastgaelam
        target_V = (adv + V_s).detach()

        # ----- Critic 更新 -----
        if self.value_clip_coef > 0.0:
            V_clipped = V_s + _clip(target_V - V_s, min=-self.value_clip_coef, max=self.value_clip_coef)
            v1 = (V_s       - target_V) * (V_s       - target_V)
            v2 = (V_clipped - target_V) * (V_clipped - target_V)
            critic_loss = paddle.mean(paddle.maximum(v1, v2))
        else:
            td = (V_s - target_V) * (V_s - target_V)
            critic_loss = paddle.mean(td)

        if not paddle.isfinite(critic_loss):
                        return paddle.to_tensor(0.0, dtype='float32')

        self._opt_critic.clear_grad(); critic_loss.backward()
        try:
            paddle.nn.utils.clip_grad_norm_(self._critic_params, max_norm=self.max_grad_norm)
        except Exception:
            pass
        self._opt_critic.step()

        # ----- Advantage 标准化 -----
        adv.stop_gradient = True
        adv = self._adv_norm(adv)

        # ----- 旧 logp（整批）-----
        i = self.agent_index
        act_i = self._norm_action_for_logprob(act_n[i], self.act_space[i])
        with paddle.no_grad():
            old_logits = self.old_policy.forward(obs_n[i])
            old_dist   = SoftPDistribution(old_logits, self.act_space[i])
            old_logp   = old_dist.log_prob(act_i)

        # ===== PPO：k 个 epoch × mini-batch 切分 =====
        idxs = np.arange(T)
        keep_num_total = 0
        keep_den_total = 0
        max_abs_logr_all = 0.0
        approx_kl_sum = 0.0
        approx_kl_cnt = 0
        last_actor_loss = 0.0
        ent_last = 0.0
        early_stop = False

        for epk in range(self.k_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, self.minibatch_size):
                mb_idx = idxs[start:start + self.minibatch_size]
                obs_mb = self._take(obs_n[i], mb_idx)
                act_mb = self._take(act_i,    mb_idx)
                adv_mb = self._take(adv,      mb_idx)
                oldlp_mb = self._take(old_logp, mb_idx)

                logits = self.model.policy(obs_mb)
                dist   = SoftPDistribution(logits, self.act_space[i])
                logp   = dist.log_prob(act_mb)

                log_ratio = paddle.clip(logp - oldlp_mb, min=-10.0, max=10.0)
                mask = paddle.abs(log_ratio) <= self.trust_clip_logr
                if mask.dtype != paddle.bool:
                    mask = mask.astype('bool')

                n_total = int(mask.shape[0])
                n_keep = int(float(paddle.sum(mask))) if n_total > 0 else 0
                keep_num_total += n_keep
                keep_den_total += n_total
                if n_keep < max(1, int(self.min_keep_ratio * n_total)):
                    continue  # 该 mini-batch 无可靠样本，跳过

                log_ratio_kept = paddle.masked_select(log_ratio, mask)
                ratio   = paddle.exp(log_ratio_kept)
                adv_kept = paddle.masked_select(adv_mb, mask)

                surr1 = ratio * adv_kept
                surr2 = _clip(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv_kept
                pg_loss = -paddle.mean(paddle.minimum(surr1, surr2))

                # 熵与 std 护栏
                try:
                    ent_all = dist.entropy(); ent_kept = paddle.masked_select(ent_all, mask); ent = paddle.mean(ent_kept)
                except Exception:
                    ent = paddle.mean(paddle.zeros_like(log_ratio_kept))
                ent_last = float(ent)

                std_reg = paddle.to_tensor(0.0, dtype=log_ratio_kept.dtype)
                if hasattr(dist, "log_std"):
                    lower, upper = -1.5, -0.3
                    over = PF.relu(dist.log_std - upper)
                    under = PF.relu(lower - dist.log_std)
                    std_reg = paddle.mean(over * over + under * under) * 5e-2

                # KL
                oldlp_kept = paddle.masked_select(oldlp_mb, mask)
                logp_kept  = oldlp_kept + log_ratio_kept
                approx_kl  = paddle.mean(oldlp_kept - logp_kept)
                approx_kl_val = float(approx_kl)
                approx_kl_sum += approx_kl_val
                approx_kl_cnt += 1
                max_abs_logr_all = max(max_abs_logr_all, float(paddle.max(paddle.abs(log_ratio_kept))))

                actor_loss = pg_loss - self.entropy_coef * ent + self.kl_coef * approx_kl + std_reg

                # 反传
                self._opt_actor.clear_grad(); actor_loss.backward()
                try:
                    paddle.nn.utils.clip_grad_norm_(self._actor_params, max_norm=self.max_grad_norm)
                except Exception:
                    pass
                self._opt_actor.step()
                last_actor_loss = float(actor_loss)

                # 自适应 KL + 早停
                if (epk + 1) >= 2 and approx_kl_val > self.target_kl * 1.5:
                    self.kl_coef = float(np.clip(self.kl_coef * 1.5, 1e-4, 10.0))
                    early_stop = True
                    break
                elif approx_kl_val < self.target_kl / 1.5:
                    self.kl_coef = float(np.clip(self.kl_coef / 1.5, 1e-4, 10.0))

            if early_stop:
                                break

        # —— old_policy 同步（按频率）—— #
        self._sync_cnt += 1
        if self._sync_cnt % self.sync_every == 0:
                        self._sync_old_policy()
        # 轻微熵衰减
        self.entropy_coef = float(np.clip(self.entropy_coef * 0.9995, 1e-4, 0.01))
        return critic_loss
