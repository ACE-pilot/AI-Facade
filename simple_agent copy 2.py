# simple_agent.py

import parl
import paddle
import numpy as np
from parl.utils import ReplayMemory

class MAAgent(parl.Agent):
    """
    多智能体 Agent，兼容连续（MADDPG/MAPPO）与离散（MADQN/QMIX）算法
    """

    def __init__(self,
                 algorithm,
                 agent_index=None,
                 obs_dim_n=None,
                 act_dim_n=None,
                 batch_size=None,
                 discrete_actions=False,
                 step_size=None):
        """
        Args:
          algorithm        (parl.core.fluid.algorithm.Algorithm or your custom Algorithm): 算法实例
          agent_index      (int): 该 Agent 在多智能体中的下标
          obs_dim_n   (list of tuple): 所有智能体观测空间形状列表
          act_dim_n     (list of int): 所有智能体动作空间维度列表
          batch_size       (int): 经验回放批次大小（仅对 replay-based 算法生效）
          discrete_actions (bool): 是否为离散动作算法（如 MADQN/QMIX）
          step_size        (float): 离散动作步长（仅在 wrapper 中使用）
        """
        # 保存新参数
        self.discrete_actions = discrete_actions
        self.step_size       = step_size

        # 验证必要参数
        assert isinstance(agent_index, int)
        assert isinstance(obs_dim_n, list)
        assert isinstance(act_dim_n, list)
        assert isinstance(batch_size, int)

        self.agent_index = agent_index
        self.obs_dim_n   = obs_dim_n
        self.act_dim_n   = act_dim_n
        self.batch_size  = batch_size
        self.n           = len(act_dim_n)

        # 初始化 replay memory（仅对需要的算法生效）
        obs_shape = obs_dim_n[self.agent_index]
        obs_dim   = int(np.prod(obs_shape))
        act_dim   = act_dim_n[self.agent_index]
        self.rpm = ReplayMemory(
            max_size=int(1e6),
            obs_dim=obs_dim,
            act_dim=act_dim
        )
        self.global_train_step = 0

        # **兼容性保护**：只有当 algorithm 是 parl.Agent 可接受的 Algorithm 时才调用 super()
        try:
            super(MAAgent, self).__init__(algorithm)
        except AssertionError:
            # 如果你的本地 MADQN/QMIX 算法不是严格的 parl.core.fluid.algorithm.Algorithm，
            # 就手动保存一份，后面直接用 self.alg 调用
            self.alg = algorithm

        # 同步一次 target（若算法提供 sync_target）
        if hasattr(self.alg, 'sync_target'):
            try:
                self.alg.sync_target(decay=0)
            except TypeError:
                self.alg.sync_target()

    def predict(self, obs):
        arr     = np.array(obs, dtype='float32').reshape(1, -1)
        tensor  = paddle.to_tensor(arr, dtype='float32')
        out     = self.alg.predict(tensor)
        return out.detach().cpu().numpy().flatten()

    def sample(self, obs, use_target_model=False):
        arr    = np.array(obs, dtype='float32').reshape(1, -1)
        tensor = paddle.to_tensor(arr, dtype='float32')
        try:
            act_out = self.alg.sample(tensor, use_target_model=use_target_model)
        except TypeError:
            act_out = self.alg.sample(tensor)
        # 离散算法返回 Q-values 向量时 argmax
        if isinstance(act_out, paddle.Tensor) and act_out.ndim == 2:
            idx = paddle.argmax(act_out, axis=1)
            return idx.cpu().numpy().flatten().tolist()
        return act_out.detach().cpu().numpy().flatten()

    def add_experience(self, obs, act, reward, next_obs, terminal):
        obs_flat  = obs.flatten()
        next_flat = next_obs.flatten()
        self.rpm.append(obs_flat, act, reward, next_flat, terminal)

    def learn(self, agents):
        self.global_train_step += 1
        if self.global_train_step % 100 != 0:
            return 0.0
        if self.rpm.size() <= self.batch_size * 25:
            return 0.0

        idxes = self.rpm.make_index(self.batch_size)
        batch_obs_n, batch_act_n, batch_obs_next_n = [], [], []
        for i in range(self.n):
            o, a, _, o2, _ = agents[i].rpm.sample_batch_by_index(idxes)
            o  = o.reshape(-1, *self.obs_dim_n[i])
            o2 = o2.reshape(-1, *self.obs_dim_n[i])
            batch_obs_n.append(o)
            batch_act_n.append(a)
            batch_obs_next_n.append(o2)

        _, _, rew_b, _, done_b = self.rpm.sample_batch_by_index(idxes)
        rew_b  = paddle.to_tensor(rew_b, dtype='float32')
        done_b = paddle.to_tensor(done_b, dtype='float32')

        batch_obs_n      = [paddle.to_tensor(x, dtype='float32') for x in batch_obs_n]
        batch_act_n      = [paddle.to_tensor(x, dtype='float32') for x in batch_act_n]
        batch_obs_next_n = [paddle.to_tensor(x, dtype='float32') for x in batch_obs_next_n]

        if hasattr(self.alg, 'Q'):
            # MADDPG / MAPPO
            target_act_next_n = [
                agents[i].alg.sample(batch_obs_next_n[i], use_target_model=True).detach()
                for i in range(self.n)
            ]
            q_next    = self.alg.Q(batch_obs_next_n, target_act_next_n, use_target_model=True)
            target_q  = rew_b + self.alg.gamma * (1.0 - done_b) * q_next.detach()
            loss      = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        else:
            # MADQN / QMIX
            loss = 0.0
            for i in range(self.n):
                obs_i  = batch_obs_n[i]
                act_i  = batch_act_n[i].astype('int64')
                next_i = batch_obs_next_n[i]
                loss   = self.alg.learn(obs_i, act_i, rew_b, next_i, done_b)

        if hasattr(loss, 'cpu'):
            return float(loss.cpu().detach().numpy())
        return float(loss)
