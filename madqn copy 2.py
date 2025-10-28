# madqn.py

import warnings
warnings.simplefilter('default')

import copy
import paddle.fluid as fluid
from parl.core.fluid.algorithm import Algorithm
from parl.core.fluid import layers

__all__ = ['MADQN']


class MADQN(Algorithm):
    def __init__(self, model, agent_index=None, act_space=None, gamma=None, lr=None,
                 use_double=False, tau=None):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.lr = lr
        self.use_double = bool(use_double)   # 是否启用 Double DQN
        self.tau = tau                       # 若给定(0,1]，则启用软更新；否则用硬同步接口

        space = act_space[agent_index]
        self.act_dim = space.n if hasattr(space, 'n') else int(space)

        # —— 在动态图下，优化器需要在初始化时绑定 parameter_list —— #
        try:
            params = self.model.parameters()              # 有些 PARL 版本
        except Exception:
            try:
                params = list(self.model.named_parameters())  # 兜底
            except Exception:
                params = None
        # 兼容老 API：AdamOptimizer / Adam
        try:
            self._opt = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, parameter_list=params)
        except Exception:
            self._opt = fluid.optimizer.Adam(learning_rate=self.lr)  # 若老版本不支持 parameter_list

    def predict(self, obs):
        return self.model.policy(obs)

    def sample(self, obs, use_target_model=False):
        net = self.target_model if use_target_model else self.model
        return net.policy(obs)

    def learn(self, obs, action, reward, next_obs, terminal, learning_rate=None):
        if learning_rate is None:
            assert isinstance(self.lr, float)
            learning_rate = self.lr
        # 动态改 lr（如果优化器支持）
        if hasattr(self._opt, 'set_lr'):
            try:
                self._opt.set_lr(learning_rate)
            except Exception:
                pass

        # 当前 Q(s,·)
        q = self.model.value(obs)              # [B, act_dim]
        # 下一步 Q_target(s',·)
        q_next_target = self.target_model.value(next_obs)  # [B, act_dim]

        if self.use_double:
            # 用 online 选 a*（避免过估计）
            q_next_online = self.model.value(next_obs)     # [B, act_dim]
            a_star = layers.argmax(q_next_online, axis=1)  # [B]
            a_star_oh = layers.one_hot(a_star, self.act_dim)
            a_star_oh = layers.cast(a_star_oh, 'float32')
            best_v = layers.reduce_sum(layers.elementwise_mul(q_next_target, a_star_oh), axis=1)
        else:
            best_v = layers.reduce_max(q_next_target, axis=1)
        best_v.stop_gradient = True

        # target
        not_done = 1.0 - layers.cast(terminal, 'float32')
        target = reward + not_done * self.gamma * best_v   # [B]

        # 选出 Q(s,a)
        act_onehot = layers.one_hot(action, self.act_dim)
        act_onehot = layers.cast(act_onehot, 'float32')
        q_sa = layers.reduce_sum(layers.elementwise_mul(q, act_onehot), axis=1)

        # Huber Loss（更稳；delta=1.0）
        td = q_sa - target
        abs_td = layers.abs(td)
        huber = layers.where(abs_td < 1.0, 0.5 * td * td, 1.0 * abs_td - 0.5)
        cost = layers.reduce_mean(huber)

        # 优化
        try:
            self._opt.minimize(cost)   # 已在 __init__ 绑定 parameter_list
        except Exception:
            # 兼容极老 API
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
        self.model.sync_weights_to(self.target_model)
