# qmix.py —— 正统版：严格要求联合动作 [B, n_agents] 或 len=n_agents 的列表
import paddle
import paddle.nn.functional as PF
import warnings
warnings.simplefilter('default')

from copy import deepcopy
from paddle import fluid
from parl.core.fluid import layers
from parl.core.fluid.algorithm import Algorithm
from gym import spaces

F = fluid.layers
__all__ = ['QMIX']

# ===== 工具封装 =====
def _one_hot(x, depth: int):
    try:
        return layers.one_hot(x, depth)
    except Exception:
        if not isinstance(x, paddle.Tensor):
            x = paddle.to_tensor(x, dtype='int64')
        elif x.dtype != paddle.int64:
            x = paddle.cast(x, 'int64')
        return PF.one_hot(x, num_classes=depth)

def _cast(x, dtype: str):
    try:
        return layers.cast(x, dtype=dtype)
    except Exception:
        return paddle.cast(x, dtype)

def _reduce_sum(x, axis=-1):
    try:
        return layers.reduce_sum(x, dim=axis)
    except Exception:
        try:
            return layers.reduce_sum(x, axis=axis)
        except Exception:
            return paddle.sum(x, axis=axis)

def _reduce_max(x, axis=-1):
    try:
        return layers.reduce_max(x, dim=axis)
    except Exception:
        try:
            return layers.reduce_max(x, axis=axis)
        except Exception:
            return paddle.max(x, axis=axis)

def _mean(x):
    try:
        return layers.reduce_mean(x)
    except Exception:
        return paddle.mean(x)

def _concat(xs, axis=-1):
    if isinstance(xs, paddle.Tensor):
        return xs
    if isinstance(xs, (list, tuple)):
        if len(xs) == 0:
            raise ValueError("_concat 收到空列表")
        if len(xs) == 1:
            x0 = xs[0]
            return x0 if isinstance(x0, paddle.Tensor) else paddle.to_tensor(x0)
        tlist = [(x if isinstance(x, paddle.Tensor) else paddle.to_tensor(x)) for x in xs]
        try:
            return layers.concat(tlist, axis=axis)
        except Exception:
            return paddle.concat(tlist, axis=axis)
    return paddle.to_tensor(xs)

def _squeeze(x, axes):
    try:
        return layers.squeeze(x, axes=axes)
    except Exception:
        return paddle.squeeze(x, axis=axes)

def _unsqueeze(x, axes):
    try:
        return layers.unsqueeze(x, axes=axes)
    except Exception:
        return paddle.unsqueeze(x, axis=axes)

def _transpose(x, perm):
    try:
        return layers.transpose(x, perm=perm)
    except Exception:
        return paddle.transpose(x, perm=perm)

def _argmax(x, axis=-1):
    try:
        return layers.argmax(x, axis=axis)
    except Exception:
        return paddle.argmax(x, axis=axis)

def _mul(x, y):
    try:
        return layers.elementwise_mul(x, y)
    except Exception:
        return paddle.multiply(x, y)

def _abs(x):
    try:
        return layers.abs(x)
    except Exception:
        return paddle.abs(x)

def _where(cond, a, b):
    try:
        return layers.where(cond, a, b)
    except Exception:
        return paddle.where(cond, a, b)

def _ensure_index_1d(idx, num_classes):
    """把动作统一成 int64 索引 [B]；支持 [B], [B,1], one-hot [B,num_classes]。"""
    if not isinstance(idx, paddle.Tensor):
        idx = paddle.to_tensor(idx)
    if idx.dtype != paddle.int64:
        idx = paddle.cast(idx, 'int64')
    if idx.ndim == 2 and idx.shape[-1] == 1:
        idx = paddle.squeeze(idx, axis=[-1])
    elif idx.ndim >= 2 and idx.shape[-1] == num_classes:
        idx = paddle.argmax(idx, axis=-1)
        if idx.dtype != paddle.int64:
            idx = paddle.cast(idx, 'int64')
    elif idx.ndim != 1:
        idx = paddle.reshape(idx, [idx.shape[0]])
    return idx

def _split_obs_to_list(x, n_agents, obs_dims=None, name="obs"):
    """把 [B,sum_obs] 拆成 n_agents 份，或校验本就是列表。"""
    if isinstance(x, (list, tuple)):
        if len(x) != n_agents:
            raise ValueError(f"{name} 列表长度应为 n_agents={n_agents}，实际 {len(x)}")
        return list(x)

    if not isinstance(x, paddle.Tensor):
        x = paddle.to_tensor(x)
    if x.ndim < 2:
        raise ValueError(f"{name} 形状至少应为 2 维 [B, ...]，实际 {list(x.shape)}")

    D = int(x.shape[-1])
    if obs_dims is not None:
        if sum(obs_dims) != D:
            raise ValueError(f"{name} 最后一维 {D} 与 sum(obs_dims)={sum(obs_dims)} 不一致")
        out, s = [], 0
        for d in obs_dims:
            out.append(x[:, s:s+d]); s += d
        return out
    else:
        if D % n_agents != 0:
            raise ValueError(f"{name} 最后一维 {D} 无法均分为 {n_agents} 份，请提供 obs_dims。")
        d = D // n_agents
        return [x[:, i*d:(i+1)*d] for i in range(n_agents)]

def _align_minibatch(obs_n, act_n, rewards, next_obs_n, dones):
    """裁到一致 B=min(...)。"""
    def _as_list(x):
        return list(x) if isinstance(x, (list, tuple)) else [x]

    obs_n      = _as_list(obs_n)
    act_n      = _as_list(act_n)
    next_obs_n = _as_list(next_obs_n)

    B = min([int((t if isinstance(t, paddle.Tensor) else paddle.to_tensor(t)).shape[0])
             for t in (obs_n + act_n + next_obs_n)])
    def _slice_first(t):
        if not isinstance(t, paddle.Tensor): t = paddle.to_tensor(t)
        return t[:B]

    obs_n      = [ _slice_first(t) for t in obs_n ]
    act_n      = [ _slice_first(t) for t in act_n ]
    next_obs_n = [ _slice_first(t) for t in next_obs_n ]
    rewards    = _slice_first(rewards)
    dones      = _slice_first(dones)
    return obs_n, act_n, rewards, next_obs_n, dones

# ===== 算法主体 =====
class QMIX(Algorithm):
    def __init__(self, model, agent_index=None, act_space=None,
                 gamma=0.99, lr=1e-3, double_q=True, tau=0.01, max_grad_norm=10.0):
        assert isinstance(agent_index, int)
        assert isinstance(act_space, list)
        self.model = model
        self.target_model = deepcopy(model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.n_agents = len(act_space)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.double_q = bool(double_q)
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        # 动作维
        self.act_dims = []
        for sp in act_space:
            if isinstance(sp, spaces.Discrete) or hasattr(sp, 'n'):
                self.act_dims.append(int(sp.n))
            else:
                raise AssertionError("当前 QMIX 实现仅支持离散动作空间。")
        self.act_dim = self.act_dims[self.agent_index]

        # 读取 per-agent obs 维度（若模型暴露）
        self.obs_dims = None
        try:
            self.obs_dims = [int(d) for d in getattr(self.model, 'agent_model').obs_shape_n]
        except Exception:
            try:
                self.obs_dims = [int(d) for d in getattr(self.model, 'obs_shape_n')]
            except Exception:
                self.obs_dims = None

        # 优化器
        params = None
        try:
            params = list(self.model.parameters())
        except Exception:
            try:
                params = [p for _, p in self.model.named_parameters()]
            except Exception:
                params = None

        grad_clip = None
        try:
            from paddle.nn import ClipGradByGlobalNorm
            grad_clip = ClipGradByGlobalNorm(self.max_grad_norm)
        except Exception:
            try:
                grad_clip = paddle.fluid.clip.GradientClipByGlobalNorm(self.max_grad_norm)
            except Exception:
                grad_clip = None

        try:
            if grad_clip is not None:
                self._opt = fluid.optimizer.AdamOptimizer(self.lr, grad_clip=grad_clip, parameter_list=params)
            else:
                self._opt = fluid.optimizer.AdamOptimizer(self.lr, parameter_list=params)
        except TypeError:
            self._opt = fluid.optimizer.AdamOptimizer(self.lr)

        self._dbg_learn_calls = 0  # 限次诊断打印

    def predict(self, obs):
        try:
            return self.model.policy(obs)
        except Exception:
            return self.model.value(obs)

    def sample(self, obs, **kwargs):
        net = self.target_model if kwargs.get('use_target_model', False) else self.model
        try:
            return net.policy(obs)
        except Exception:
            return net.value(obs)

    def _soft_update(self, src, dst, tau):
        s, d = dict(src.named_parameters()), dict(dst.named_parameters())
        for k in d:
            if k in s:
                d[k].set_value(tau * s[k] + (1.0 - tau) * d[k])

    def sync_target(self):
        if hasattr(self.model, 'sync_weights_to'):
            self.model.sync_weights_to(self.target_model)
        else:
            s, d = dict(self.model.named_parameters()), dict(self.target_model.named_parameters())
            for k in d:
                if k in s:
                    d[k].set_value(s[k])

    def learn(self, obs_n, act_n, rewards, next_obs_n, dones, states=None, next_states=None, learning_rate=None):
        # 动态 lr
        if learning_rate is None:
            learning_rate = self.lr
        if hasattr(self._opt, 'set_lr'):
            try: self._opt.set_lr(learning_rate)
            except Exception: pass

        # 限次诊断打印
        if self._dbg_learn_calls < 5:
            def _shape(x):
                if isinstance(x, (list, tuple)):
                    return [list(getattr(t, 'shape', ())) if hasattr(t, 'shape') else type(t) for t in x]
                return list(getattr(x, 'shape', ())) if hasattr(x, 'shape') else type(x)
            print(f"[QMIX.learn dbg#{self._dbg_learn_calls}] "
                f"obs_n={_shape(obs_n)}, act_n={_shape(act_n)}, "
                f"rewards={_shape(rewards)}, next_obs_n={_shape(next_obs_n)}, dones={_shape(dones)}")
            self._dbg_learn_calls += 1

        # 拆分 obs
        try:
            obs_n      = _split_obs_to_list(obs_n,      self.n_agents, self.obs_dims, name="obs_n")
            next_obs_n = _split_obs_to_list(next_obs_n, self.n_agents, self.obs_dims, name="next_obs_n")
        except Exception as e:
            raise RuntimeError(f"QMIX.learn: 无法解析 per-agent 观测。{e}")

        # —— 严格规范化动作：只能是列表(len=n_agents)或 Tensor[B,n_agents] —— #
        if isinstance(act_n, (list, tuple)):
            if len(act_n) != self.n_agents:
                raise RuntimeError(
                    f"QMIX.learn: 期望 act_n 为长度 {self.n_agents} 的列表（每个元素 [B] / [B,1] / one-hot），实际 {len(act_n)}。"
                )
            act_list = [ _ensure_index_1d(a, self.act_dims[i]) for i, a in enumerate(act_n) ]
        else:
            if not isinstance(act_n, paddle.Tensor):
                act_n = paddle.to_tensor(act_n)
            if act_n.ndim != 2 or int(act_n.shape[1]) != self.n_agents:
                head = act_n[:3].numpy() if hasattr(act_n, 'numpy') else None
                raise RuntimeError(
                    f"QMIX.learn: 期望 act_n 形状为 [B, {self.n_agents}] 的联合动作，实际 {list(act_n.shape)}。\n"
                    f"请在外层把每步动作组装为联合动作（如 np.array([a0,...,a{self.n_agents-1}], int64)），"
                    f"或改为传入长度为 {self.n_agents} 的动作列表。\n"
                    f"act_n 头部（前 3 行）: {head}"
                )
            act_list = [ _ensure_index_1d(act_n[:, i], self.act_dims[i]) for i in range(self.n_agents) ]
        act_n = act_list

        # 对齐 batch
        obs_n, act_n, rewards, next_obs_n, dones = _align_minibatch(
            obs_n, act_n, rewards, next_obs_n, dones
        )

        # states
        B = int(obs_n[0].shape[0])
        if states is None:
            states = _concat(obs_n, axis=1)  # [B, sum_obs]
        if next_states is None:
            next_states = _concat(next_obs_n, axis=1)
        if not isinstance(states, paddle.Tensor): states = paddle.to_tensor(states)
        if not isinstance(next_states, paddle.Tensor): next_states = paddle.to_tensor(next_states)
        states = _cast(states, 'float32')[:B]
        next_states = _cast(next_states, 'float32')[:B]

        # 选中每体 Q(o_i, a_i) -> [B, n_agents, 1]
        chosen_qs = []
        for i in range(self.n_agents):
            q_i = self.model.value(obs_n[i])                    # [B, A_i]
            a_i = _ensure_index_1d(act_n[i], self.act_dims[i])  # [B]
            oh  = _cast(_one_hot(a_i, self.act_dims[i]), 'float32')
            q_i_a = _reduce_sum(_mul(q_i, oh), axis=1)          # [B]
            q_i_a = _unsqueeze(_unsqueeze(q_i_a, axes=[-1]), axes=[-1])  # [B,1,1]
            chosen_qs.append(q_i_a)
        chosen_qs = _concat(chosen_qs, axis=1)                  # [B, n_agents, 1]

        # 当前 Q_tot
        q_tot = self.model.mix(chosen_qs, states)               # [B,1] 或 [B]
        if isinstance(q_tot, paddle.Tensor) and q_tot.ndim > 1 and int(q_tot.shape[-1]) == 1:
            q_tot = _squeeze(q_tot, axes=[-1])                  # [B]

        # 目标 Q_tot
        next_qs_tar = []
        if self.double_q:
            next_qs_online = [self.model.value(next_obs_n[i]) for i in range(self.n_agents)]  # [B,A_i]
            next_acts_star = [_argmax(q, axis=1) for q in next_qs_online]                     # [B]
            for i in range(self.n_agents):
                q_tar = self.target_model.value(next_obs_n[i])                                # [B,A_i]
                idx_star = _ensure_index_1d(next_acts_star[i], self.act_dims[i])              # [B]
                oh = _cast(_one_hot(idx_star, self.act_dims[i]), 'float32')
                q_i_a = _reduce_sum(_mul(q_tar, oh), axis=1)
                q_i_a = _unsqueeze(_unsqueeze(q_i_a, axes=[-1]), axes=[-1])                   # [B,1,1]
                next_qs_tar.append(q_i_a)
        else:
            for i in range(self.n_agents):
                q_tar = self.target_model.value(next_obs_n[i])
                q_i_max = _reduce_max(q_tar, axis=1)
                q_i_max = _unsqueeze(_unsqueeze(q_i_max, axes=[-1]), axes=[-1])               # [B,1,1]
                next_qs_tar.append(q_i_max)

        next_qs_tar = _concat(next_qs_tar, axis=1)                                            # [B, n_agents, 1]
        q_tot_next = self.target_model.mix(next_qs_tar, next_states)          # [B,1] 或 [B]
        if isinstance(q_tot_next, paddle.Tensor) and q_tot_next.ndim > 1 and int(q_tot_next.shape[-1]) == 1:
            q_tot_next = _squeeze(q_tot_next, axes=[-1])                      # [B]

        # TD(Huber) 损失 —— 先 cast 成 Tensor，再按维度 squeeze（不再使用 F.shape）
        rewards = _cast(rewards, 'float32')
        dones   = _cast(dones,   'float32')
        if isinstance(rewards, paddle.Tensor) and rewards.ndim > 1:
            rewards = _squeeze(rewards, axes=[-1])
        if isinstance(dones, paddle.Tensor) and dones.ndim > 1:
            dones = _squeeze(dones, axes=[-1])

        target_tot = rewards + self.gamma * (1.0 - dones) * q_tot_next

        td = q_tot - target_tot
        abs_td = _abs(td)
        huber = _where(abs_td < 1.0, 0.5 * td * td, abs_td - 0.5)
        loss = _mean(huber)

        self._opt.minimize(loss)
        if isinstance(self.tau, float) and 0.0 < self.tau <= 1.0:
            self._soft_update(self.model, self.target_model, self.tau)

        return loss

