
"""
问题的根因就是：在动态图下必须用 loss.backward() → opt.step() → opt.clear_grad() 的更新路径；
改掉 fluid.optimizer.minimize 之后参数终于真的在动了，因此训练正常起势。
"""


import warnings
warnings.simplefilter('default')

import copy
import paddle
import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers

__all__ = ['MADQN']

F = layers  # 简写，便于阅读

# ===================== 工具函数（兼容 Paddle 2.x / fluid） =====================

def _cast(x, dtype: str):
    try:
        return layers.cast(x, dtype=dtype)
    except Exception:
        return paddle.cast(x, dtype)

def _one_hot(x, depth: int):
    """兼容 fluid.layers.one_hot 与 paddle.nn.functional.one_hot"""
    try:
        return layers.one_hot(x, depth)
    except Exception:
        if not isinstance(x, paddle.Tensor):
            x = paddle.to_tensor(x, dtype='int64')
        elif x.dtype != paddle.int64:
            x = paddle.cast(x, 'int64')
        return paddle.nn.functional.one_hot(x, num_classes=depth)

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
    """
    把各种形式的动作统一成 1D 的 int64 索引 [B]：
      - 若是 [B,1] -> squeeze 到 [B]
      - 若是 one-hot [B, num_classes] -> argmax 到 [B]
      - 若是其它 >=2 维 -> 尝试 reshape 到 [B]
    """
    if not isinstance(idx, paddle.Tensor):
        idx = paddle.to_tensor(idx)
    if idx.dtype != paddle.int64:
        idx = paddle.cast(idx, 'int64')
    # [B, 1] → [B]
    if idx.ndim == 2 and int(idx.shape[-1]) == 1:
        idx = paddle.squeeze(idx, axis=[-1])
    # one-hot [B, A] → argmax [B]
    elif idx.ndim >= 2 and int(idx.shape[-1]) == int(num_classes):
        idx = paddle.argmax(idx, axis=-1)
        if idx.dtype != paddle.int64:
            idx = paddle.cast(idx, 'int64')
    # 其它多维，尽量压成 [B]
    elif idx.ndim != 1:
        idx = paddle.reshape(idx, [idx.shape[0]])
    return idx

# ===================== MADQN 定义 =====================

class MADQN(Algorithm):
    """
    多智能体 DQN（独立学习版本）：
      - 默认启用 Double DQN（use_double=True）
      - 支持 tau 软更新（tau in (0,1]）或硬同步（sync_target）
      - Q 网络接口优先使用 model.policy(obs)，若不存在则退回 model.value(obs)
    说明：
      - 探索策略（epsilon-greedy）请在 Agent 层实现（已在 simple_agent.py 中给出）
      - 回放中的动作应存索引（int64），本类会用 one-hot 选取 Q(s,a)
    """

    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=0.99,
                 lr=3e-4,
                 use_double=True,
                 tau=0.01):
        # ★ 关键修复：把 model 传给父类，避免 AssertionError
        super(MADQN, self).__init__(model)

        self.model = model
        self.target_model = copy.deepcopy(model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.use_double = bool(use_double)   # 默认开启 Double DQN
        self.tau = tau if (tau is None or (0.0 < float(tau) <= 1.0)) else 0.01

        space = act_space[agent_index]
        self.act_dim = space.n if hasattr(space, 'n') else int(space)

        # —— 动态图优化器 —— #
        try:
            params = list(self.model.parameters())
        except Exception:
            try:
                params = [p for _, p in self.model.named_parameters()]
            except Exception:
                params = None

        # —— 动态图优先；静态图兜底 —— #
        self._use_paddle_opt = False
        try:
            if paddle.in_dynamic_mode():
                # 动态图正确写法
                self._opt = paddle.optimizer.Adam(learning_rate=self.lr, parameters=params)
                self._use_paddle_opt = True
            else:
                # 静态图（老 PARL/fluid）写法
                self._opt = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, parameter_list=params)
        except Exception:
            # 兜底：再试一次 fluid.Adam
            try:
                self._opt = fluid.optimizer.Adam(learning_rate=self.lr, parameter_list=params)
            except Exception:
                # 最后兜底（极老版本）
                self._opt = fluid.optimizer.Adam(learning_rate=self.lr)


        self._dbg = 0  # 仅前几次学习打印调试信息

    # —— 统一取 Q(o,·)：优先 policy(obs)，失败退 value(obs) —— #
    def _qvec(self, net, obs):
        try:
            return net.policy(obs)   # 多数离散模型这就是 Q/logits，形状 [B, A]
        except Exception:
            return net.value(obs)    # 少数模型把 Q 放在 value(obs)

    def predict(self, obs):
        """前向（评估用）：返回 Q(s,·)"""
        return self._qvec(self.model, obs)

    def sample(self, obs, use_target_model=False):
        """前向（训练用）：返回 Q(s,·)。探索在 Agent 里做，这里不做 epsilon。"""
        net = self.target_model if use_target_model else self.model
        return self._qvec(net, obs)

    def learn(self, obs, action, reward, next_obs, terminal, learning_rate=None):
        """
        obs:       [B, obs_dim]
        action:    索引 [B] 或 one-hot [B, A] 或 [B,1]（内部统一成 [B]）
        reward:    [B] 或 [B,1]
        next_obs:  [B, obs_dim]
        terminal:  [B] 或 [B,1]，1.0 表示 done
        """
        # —— 学习率（可选动态调整） —— #
        if learning_rate is None:
            learning_rate = self.lr
        # 动态图 optimizer 支持 set_lr；静态图不一定有
        if hasattr(self._opt, 'set_lr'):
            try:
                self._opt.set_lr(learning_rate)
            except Exception:
                pass

        # —— 统一 dtype / shape —— #
        if not isinstance(obs, paddle.Tensor):
            obs = paddle.to_tensor(obs, dtype='float32')
        if not isinstance(next_obs, paddle.Tensor):
            next_obs = paddle.to_tensor(next_obs, dtype='float32')
        if not isinstance(reward, paddle.Tensor):
            reward = paddle.to_tensor(reward, dtype='float32')
        if not isinstance(terminal, paddle.Tensor):
            terminal = paddle.to_tensor(terminal, dtype='float32')

        # squeeze 奖励/终止到 [B]
        if reward.ndim > 1:
            reward = paddle.squeeze(reward, axis=[-1])
        if terminal.ndim > 1:
            terminal = paddle.squeeze(terminal, axis=[-1])

        # 动作标准化为 [B] 的 int64 索引
        action = _ensure_index_1d(action, self.act_dim)  # [B] int64

        # —— 前向：Q(s,·)、Q_target(s',·) —— #
        q = self._qvec(self.model, obs)                         # [B, A]
        q_next_target = self._qvec(self.target_model, next_obs) # [B, A]

        # —— 目标值（Double DQN 可选）—— #
        if self.use_double:
            q_next_online = self._qvec(self.model, next_obs)        # [B, A]
            a_star = _argmax(q_next_online, axis=1)                 # [B]
            a_star_oh = _cast(_one_hot(a_star, self.act_dim), 'float32')
            best_v = _reduce_sum(_mul(q_next_target, a_star_oh), axis=1)  # [B]
        else:
            best_v = _reduce_max(q_next_target, axis=1)             # [B]
        best_v.stop_gradient = True

        not_done = 1.0 - _cast(terminal, 'float32')
        target = reward + self.gamma * not_done * best_v            # [B]

        # —— 取 Q(s,a) —— #
        act_onehot = _cast(_one_hot(action, self.act_dim), 'float32')  # [B, A]
        q_sa = _reduce_sum(_mul(q, act_onehot), axis=1)                 # [B]

        # —— Huber Loss（δ=1.0）—— #
        td = q_sa - target
        abs_td = _abs(td)
        huber = _where(abs_td < 1.0, 0.5 * td * td, abs_td - 0.5)
        cost = paddle.mean(huber)

        # —— 调试信息（仅前 3 次）—— #
        if not hasattr(self, "_dbg_print"):
            self._dbg_print = 0
        if self._dbg_print < 3:
            print("[MADQN.learn] batch:",
                f"obs={list(obs.shape)} act={list(action.shape)} "
                f"rew={list(reward.shape)} next={list(next_obs.shape)} done={list(terminal.shape)}")
            try:
                print("  q.mean=%.4f target.mean=%.4f loss=%.4f" %
                    (float(paddle.mean(q)), float(paddle.mean(target)), float(cost)))
            except Exception:
                pass
            self._dbg_print += 1

        # —— 训练模式 —— #
        try:
            self.model.train()
            self.target_model.train()
        except Exception:
            pass

        # ——（可选）参数均值变化观察，仅前 3 次 —— #
        if not hasattr(self, "_dbg_upd"):
            self._dbg_upd = 0
        before = None
        pname0 = None
        if self._dbg_upd < 3:
            try:
                pname0, p0 = next(iter(self.model.named_parameters()))
                before = float(paddle.mean(p0))
            except Exception:
                before = None

        # —— 反传 + 更新（动态图优先）—— #
        use_paddle_opt = bool(getattr(self, "_use_paddle_opt", paddle.in_dynamic_mode()))
        if use_paddle_opt:
            cost.backward()
            # （可选）梯度裁剪：
            # paddle.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self._opt.step()
            self._opt.clear_grad()
        else:
            # 静态图兜底
            try:
                self._opt.minimize(cost)
            except Exception:
                optimizer = fluid.optimizer.Adam(learning_rate=float(learning_rate), epsilon=1e-3)
                optimizer.minimize(cost)

        # —— 参数均值对比 —— #
        if self._dbg_upd < 3 and before is not None:
            try:
                pname1, p1 = next(iter(self.model.named_parameters()))
                after = float(paddle.mean(p1))
            except Exception:
                pass
            self._dbg_upd += 1

        # —— 软更新（若启用）—— #
        if isinstance(self.tau, float) and 0.0 < self.tau <= 1.0:
            self._soft_update(self.model, self.target_model, self.tau)

        return cost

    

    # ===================== Target 网络更新 =====================

    def _soft_update(self, src_model, dst_model, tau):
        """θ_target ← τ·θ + (1-τ)·θ_target"""
        src = dict(src_model.named_parameters())
        dst = dict(dst_model.named_parameters())
        for k in dst:
            if k in src:
                dst[k].set_value(tau * src[k] + (1.0 - tau) * dst[k])

    def sync_target(self):
        """硬同步 target 网络"""
        if hasattr(self.model, 'sync_weights_to'):
            self.model.sync_weights_to(self.target_model)
        else:
            src = dict(self.model.named_parameters())
            dst = dict(self.target_model.named_parameters())
            for k in dst:
                if k in src:
                    dst[k].set_value(src[k])
