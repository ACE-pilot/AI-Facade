# madqn.py
import warnings
warnings.simplefilter('default')

import copy
import paddle
import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers

__all__ = ['MADQN']

F = layers  # 便于阅读

# ==== 通用小工具（与 qmix 同风格，保证在 paddle 2.x 下稳） ====
def _cast(x, dtype: str):
    try:
        return layers.cast(x, dtype=dtype)
    except Exception:
        return paddle.cast(x, dtype)

def _one_hot(x, depth: int):
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
    if idx.ndim == 2 and int(idx.shape[-1]) == 1:
        idx = paddle.squeeze(idx, axis=[-1])
    elif idx.ndim >= 2 and int(idx.shape[-1]) == int(num_classes):
        idx = paddle.argmax(idx, axis=-1)
        if idx.dtype != paddle.int64:
            idx = paddle.cast(idx, 'int64')
    elif idx.ndim != 1:
        idx = paddle.reshape(idx, [idx.shape[0]])
    return idx

class MADQN(Algorithm):
    def __init__(self, model, agent_index=None, act_space=None, gamma=None, lr=None,
                 use_double=False, tau=None):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.use_double = bool(use_double)   # 是否启用 Double DQN
        self.tau = tau                       # 若给定(0,1]，则启用软更新；否则用硬同步接口

        space = act_space[agent_index]
        self.act_dim = space.n if hasattr(space, 'n') else int(space)

        # —— 在动态图下，优化器需要在初始化时绑定 parameter_list —— #
        try:
            params = list(self.model.parameters())
        except Exception:
            try:
                params = [p for _, p in self.model.named_parameters()]
            except Exception:
                params = None
        # 兼容老 API：AdamOptimizer / Adam
        try:
            self._opt = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, parameter_list=params)
        except Exception:
            self._opt = fluid.optimizer.Adam(learning_rate=self.lr)  # 若老版本不支持 parameter_list

        self._dbg = 0  # 限次打印

    # —— 统一取 Q(o,·)：优先 policy(obs)，失败退 value(obs) —— #
    def _qvec(self, net, obs):
        try:
            return net.policy(obs)   # 多数离散模型这就是 Q/logits，形状 [B, A]
        except Exception:
            return net.value(obs)    # 少数模型把 Q 放在 value(obs)

    def predict(self, obs):
        return self._qvec(self.model, obs)

    def sample(self, obs, use_target_model=False):
        net = self.target_model if use_target_model else self.model
        return self._qvec(net, obs)

    def learn(self, obs, action, reward, next_obs, terminal, learning_rate=None):
        # 设置/更新 lr
        if learning_rate is None:
            learning_rate = self.lr
        if hasattr(self._opt, 'set_lr'):
            try:
                self._opt.set_lr(learning_rate)
            except Exception:
                pass

        # —— 统一 dtype/shape —— #
        if not isinstance(obs, paddle.Tensor):       obs       = paddle.to_tensor(obs, dtype='float32')
        if not isinstance(next_obs, paddle.Tensor):  next_obs  = paddle.to_tensor(next_obs, dtype='float32')
        if not isinstance(reward, paddle.Tensor):    reward    = paddle.to_tensor(reward, dtype='float32')
        if not isinstance(terminal, paddle.Tensor):  terminal  = paddle.to_tensor(terminal, dtype='float32')
        action = _ensure_index_1d(action, self.act_dim)        # [B] int64

        # squeeze 奖励/终止到 [B]
        if reward.ndim  > 1: reward  = paddle.squeeze(reward,  axis=[-1])
        if terminal.ndim > 1: terminal = paddle.squeeze(terminal, axis=[-1])

        # 当前 Q(s,·) 与 target Q(s',·)
        q = self._qvec(self.model, obs)                      # [B, A]
        q_next_target = self._qvec(self.target_model, next_obs)  # [B, A]

        # Double-DQN 或 Max
        if self.use_double:
            q_next_online = self._qvec(self.model, next_obs)     # [B, A]
            a_star = _argmax(q_next_online, axis=1)              # [B]
            a_star_oh = _cast(_one_hot(a_star, self.act_dim), 'float32')
            best_v = _reduce_sum(_mul(q_next_target, a_star_oh), axis=1)  # [B]
        else:
            best_v = _reduce_max(q_next_target, axis=1)          # [B]
        best_v.stop_gradient = True

        # TD target
        not_done = 1.0 - _cast(terminal, 'float32')
        target = reward + self.gamma * not_done * best_v         # [B]

        # 选出 Q(s,a)
        act_onehot = _cast(_one_hot(action, self.act_dim), 'float32')   # [B, A]
        q_sa = _reduce_sum(_mul(q, act_onehot), axis=1)                  # [B]

        # Huber Loss（delta=1.0）
        td = q_sa - target
        abs_td = _abs(td)
        huber = _where(abs_td < 1.0, 0.5 * td * td, abs_td - 0.5)
        cost = paddle.mean(huber)

        # 限次诊断打印（仅前 3 次）
        if self._dbg < 3:
            def shp(t): return list(t.shape) if isinstance(t, paddle.Tensor) else type(t)
            print(f"[MADQN.learn dbg#{self._dbg}] q={shp(q)}, act=[{int(self.act_dim)}], "
                  f"obs={shp(obs)}, next={shp(next_obs)}, rew={shp(reward)}, done={shp(terminal)}")
            self._dbg += 1

        # 反传
        try:
            self._opt.minimize(cost)
        except Exception:
            optimizer = fluid.optimizer.Adam(learning_rate=learning_rate, epsilon=1e-3)
            optimizer.minimize(cost)

        # 软更新（若 tau 给定），否则外部定期调用 sync_target()
        if isinstance(self.tau, float) and 0.0 < self.tau <= 1.0:
            self._soft_update(self.model, self.target_model, self.tau)

        return cost

    def _soft_update(self, src_model, dst_model, tau):
        # 逐参数：θ_target ← τ θ + (1-τ) θ_target
        src = dict(src_model.named_parameters())
        dst = dict(dst_model.named_parameters())
        for k in dst:
            if k in src:
                dst[k].set_value(tau * src[k] + (1.0 - tau) * dst[k])

    def sync_target(self):
        # 某些 PARL 版本 Model 自带该接口
        if hasattr(self.model, 'sync_weights_to'):
            self.model.sync_weights_to(self.target_model)
        else:
            # 兜底：硬同步
            src = dict(self.model.named_parameters())
            dst = dict(self.target_model.named_parameters())
            for k in dst:
                if k in src:
                    dst[k].set_value(src[k])
