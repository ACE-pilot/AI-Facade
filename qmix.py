# qmix.py —— 正统版：严格要求联合动作 [B, n_agents] 或 len=n_agents 的列表
import paddle
import paddle as _pd
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
        self._dbg_learn_calls = 0  # 限次诊断打印

        # --- DEBUG helper: 统计 Mixer 参数 L2（供 pre/post 打印） ---
        def _mixer_l2(model_obj):
            try:
                tot = 0.0
                for p in getattr(model_obj, 'mixer').parameters():
                    v = p.detach().cpu().numpy()
                    tot += float((v * v).sum())
                return tot
            except Exception:
                return None
        self._mixer_l2 = _mixer_l2

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

        # >>> 强制构建 QMixer 参数（否则优化器看不到懒初始化出来的权重）
        try:
            if hasattr(self.model, "mixer"):
                if not getattr(self.model.mixer, "_params_built", False):
                    state_dim = sum(self.obs_dims) if self.obs_dims else None
                    assert state_dim is not None, "无法推断 state_dim 用于 QMixer 初始化"
                    self.model.mixer._build_params(state_dim)
            if hasattr(self.target_model, "mixer"):
                if not getattr(self.target_model.mixer, "_params_built", False):
                    state_dim = sum(self.obs_dims) if self.obs_dims else None
                    self.target_model.mixer._build_params(state_dim)
        except Exception:
            pass
        # <<< 强制构建


        # 优化器 —— 牢靠收集参数（兼容 parl.Model + paddle.nn.Layer 的组合）
        params = []
        _seen = set()

        def _add_params(obj):
            if obj is None:
                return
            try:
                ps = list(obj.parameters())
            except Exception:
                try:
                    ps = [p for _, p in obj.named_parameters()]
                except Exception:
                    ps = []
            for p in ps:
                k = id(p)
                if k not in _seen:
                    _seen.add(k)
                    params.append(p)

        # 先加复合模型，再兜底各子模块
        _add_params(self.model)
        _add_params(getattr(self.model, 'agent_model', None))
        _add_params(getattr(self.model, 'mixer', None))

        # 显式允许 mixer 参与反传（即便单体直通时它也不会被用到）
        if hasattr(self.model, 'mixer'):
            try:
                for p in self.model.mixer.parameters():
                    p.stop_gradient = False
            except Exception:
                pass

        if not params:
            raise RuntimeError("QMIX: 未收集到任何可训练参数（请检查模型装配）。")

        # 梯度裁剪（可选）
        grad_clip = None
        try:
            from paddle.nn import ClipGradByGlobalNorm
            grad_clip = ClipGradByGlobalNorm(self.max_grad_norm)
        except Exception:
            try:
                grad_clip = paddle.fluid.clip.GradientClipByGlobalNorm(self.max_grad_norm)
            except Exception:
                grad_clip = None

        # 动态图优化器优先
        try:
            from paddle.optimizer import Adam as _Adam
            if grad_clip is not None:
                self._opt = _Adam(learning_rate=self.lr, parameters=params, grad_clip=grad_clip)
            else:
                self._opt = _Adam(learning_rate=self.lr, parameters=params)
        except Exception:
            if grad_clip is not None:
                self._opt = fluid.optimizer.AdamOptimizer(self.lr, grad_clip=grad_clip, parameter_list=params)
            else:
                self._opt = fluid.optimizer.AdamOptimizer(self.lr, parameter_list=params)

        # 一次性打印参数计数，确认 agent_model/mixer 确实进了优化器
        if not hasattr(self, "_dbg_param_printed"):
            try:
                def _numel(p):
                    s = getattr(p, 'shape', None)
                    if s is None: return 0
                    n = 1
                    for d in s: n *= int(d)
                    return n
                tot = sum(_numel(p) for p in params)
                mix = 0
                if hasattr(self.model, 'mixer'):
                    try:
                        mix = sum(_numel(p) for p in self.model.mixer.parameters())
                    except Exception:
                        pass
                agq = 0
                if hasattr(self.model, 'agent_model'):
                    try:
                        agq = sum(_numel(p) for p in self.model.agent_model.parameters())
                    except Exception:
                        pass
                print(f"[QMIX.opt] params_total={tot}, agent_Q_params={agq}, mixer_params={mix}")
            except Exception:
                pass
            self._dbg_param_printed = True



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
        import paddle
        with paddle.no_grad():
            for k in d:
                if k in s:
                    new_v = s[k] * tau + d[k] * (1.0 - tau)
                    d[k].set_value(new_v.detach())

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
            if not isinstance(act_n, _pd.Tensor):
                act_n = _pd.to_tensor(act_n)
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
            
        # ---- 单体特例：退化为标准 DQN（稳定且更易起势）----
        if self.n_agents == 1:
            # 选中 Q(s,a)
            q_now = self.model.value(obs_n[0])                    # [B, A]
            a_idx = _ensure_index_1d(act_n[0], self.act_dims[0])  # [B]
            oh    = _cast(_one_hot(a_idx, self.act_dims[0]), 'float32')
            q_chosen = _reduce_sum(_mul(q_now, oh), axis=1)       # [B]

            # 目标 Q
            if self.double_q:
                q_next_online = self.model.value(next_obs_n[0])   # [B, A]
                a_star = _argmax(q_next_online, axis=1)           # [B]
                q_next_tar = self.target_model.value(next_obs_n[0])          # [B, A]
                oh2 = _cast(_one_hot(_ensure_index_1d(a_star, self.act_dims[0]), self.act_dims[0]), 'float32')
                q_next = _reduce_sum(_mul(q_next_tar, oh2), axis=1)          # [B]
            else:
                q_next_tar = self.target_model.value(next_obs_n[0])          # [B, A]
                q_next = _reduce_max(q_next_tar, axis=1)                     # [B]

            # TD(Huber) 损失
            rewards = _cast(rewards, 'float32')
            dones   = _cast(dones,   'float32')
            if isinstance(rewards, _pd.Tensor) and rewards.ndim > 1: rewards = _squeeze(rewards, axes=[-1])
            if isinstance(dones,   _pd.Tensor) and dones.ndim   > 1: dones   = _squeeze(dones,   axes=[-1])

            target = rewards + self.gamma * (1.0 - dones) * q_next
            td     = q_chosen - target
            abs_td = _abs(td)
            huber  = _where(abs_td < 1.0, 0.5 * td * td, abs_td - 0.5)
            loss   = _mean(huber)

            # === 与动态图分支一致的更新 ===
            if hasattr(self._opt, "step") and hasattr(self._opt, "clear_grad"):
                self.model.train()
                try: self._opt.clear_grad()
                except Exception: pass
                loss.backward()

                # 仅前 3 次：打印个体 Q 的梯度强度
                if getattr(self, "_dbg_grad_prints", 0) < 3:
                    try:
                        g2_q = 0.0
                        if hasattr(self.model, 'agent_model'):
                            for p in self.model.agent_model.parameters():
                                if getattr(p, "grad", None) is not None:
                                    g2_q += float((p.grad * p.grad).sum())
                        print(f"[QMIX.grad-DQN] agent_Q_grad_L2={g2_q:.6e}")
                    except Exception:
                        pass
                    self._dbg_grad_prints = getattr(self, "_dbg_grad_prints", 0) + 1

                try:
                    from paddle.nn.utils import clip_grad_norm_
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                except Exception:
                    pass
                self._opt.step()
            else:
                try:
                    self._opt.minimize(loss)
                except Exception as e:
                    raise RuntimeError(f"QMIX.learn(single-agent): minimize failed: {e}")

            # 软更新 target
            if isinstance(self.tau, float) and 0.0 < self.tau <= 1.0:
                self._soft_update(self.model, self.target_model, self.tau)

            return loss

        
        if not isinstance(states, _pd.Tensor): states = _pd.to_tensor(states)
        if not isinstance(next_states, _pd.Tensor): next_states = _pd.to_tensor(next_states)
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
        if isinstance(q_tot, _pd.Tensor) and q_tot.ndim > 1 and int(q_tot.shape[-1]) == 1:
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
        if isinstance(q_tot_next, _pd.Tensor) and q_tot_next.ndim > 1 and int(q_tot_next.shape[-1]) == 1:
            q_tot_next = _squeeze(q_tot_next, axes=[-1])                      # [B]

        # TD(Huber) 损失 —— 先 cast 成 Tensor，再按维度 squeeze（不再使用 F.shape）
        rewards = _cast(rewards, 'float32')
        dones   = _cast(dones,   'float32')
        if isinstance(rewards, _pd.Tensor) and rewards.ndim > 1:
            rewards = _squeeze(rewards, axes=[-1])
        if isinstance(dones, _pd.Tensor) and dones.ndim > 1:
            dones = _squeeze(dones, axes=[-1])

        target_tot = rewards + self.gamma * (1.0 - dones) * q_tot_next

        td = q_tot - target_tot
        abs_td = _abs(td)
        huber = _where(abs_td < 1.0, 0.5 * td * td, abs_td - 0.5)

        # --- L2 pre (限前5次) ---
        if self._dbg_learn_calls < 5:
            try:
                l2_m = self._mixer_l2(self.model)
                l2_t = self._mixer_l2(self.target_model)
                print(f"[QMIX.learn L2 pre] model={l2_m:.6f}  target={l2_t:.6f}")
            except Exception:
                pass

        loss = _mean(huber)

        # --- Update path (动态图优先，fluid兜底) ---
        if hasattr(self._opt, "step") and hasattr(self._opt, "clear_grad"):
            # Dygraph optimizer
            self.model.train()
            try:
                self._opt.clear_grad()
            except Exception:
                pass
            loss.backward()

            # === DEBUG: 仅前3次打印梯度 L2，确认 agent Q 网络在学（单体直通时 mixer 梯度为 0 属正常） ===
            if getattr(self, "_dbg_grad_prints", 0) < 3:
                try:
                    g2_q = 0.0
                    if hasattr(self.model, 'agent_model'):
                        for p in self.model.agent_model.parameters():
                            if getattr(p, "grad", None) is not None:
                                g2_q += float((p.grad * p.grad).sum())
                    g2_m = 0.0
                    if hasattr(self.model, 'mixer'):
                        for p in self.model.mixer.parameters():
                            if getattr(p, "grad", None) is not None:
                                g2_m += float((p.grad * p.grad).sum())
                    print(f"[QMIX.grad] agent_Q_grad_L2={g2_q:.6e}  mixer_grad_L2={g2_m:.6e}")
                except Exception:
                    pass
                self._dbg_grad_prints = getattr(self, "_dbg_grad_prints", 0) + 1

            try:
                from paddle.nn.utils import clip_grad_norm_
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            except Exception:
                pass
            self._opt.step()


            # 一次性确认走了动态图分支
            if not hasattr(self, "_dbg_opt_printed"):
                print("[QMIX.opt] use dygraph step/clear_grad")
                self._dbg_opt_printed = True
        else:
            # Fluid optimizer（内部会做 backward）
            try:
                self._opt.minimize(loss)
            except Exception as e:
                raise RuntimeError(f"QMIX.learn: fluid minimize failed: {e}")

            # 一次性确认走了fluid分支
            if not hasattr(self, "_dbg_opt_printed"):
                print("[QMIX.opt] use fluid minimize")
                self._dbg_opt_printed = True

        if isinstance(self.tau, float) and 0.0 < self.tau <= 1.0:
            self._soft_update(self.model, self.target_model, self.tau)

        # --- L2 post (限前5次) ---
        if self._dbg_learn_calls < 5:
            try:
                l2_m2 = self._mixer_l2(self.model)
                l2_t2 = self._mixer_l2(self.target_model)
                print(f"[QMIX.learn L2 post] model={l2_m2:.6f} target={l2_t2:.6f} "
                      f"Δmodel={l2_m2-l2_m:+.3e} Δtarget={l2_t2-l2_t:+.3e}")
            except Exception:
                pass
            self._dbg_learn_calls += 1

        return loss
