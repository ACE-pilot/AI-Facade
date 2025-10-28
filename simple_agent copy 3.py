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

        # === QMIX PATCH: 供 train.py 注入集中式回放与身份标记（默认无） ===
        self.shared_rpm = None
        self.algo_name  = None
        # === END QMIX PATCH ===

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
        # === QMIX PATCH: 仅在 QMIX 时，从“集中式回放”采样并传联合动作给算法 ===
        if self.algo_name == 'qmix' and self.shared_rpm is not None:
            self.global_train_step += 1
            if self.global_train_step % 100 != 0:
                return 0.0
            if self.shared_rpm.size() <= self.batch_size * 25:
                return 0.0

            batch = self.shared_rpm.sample_batch(self.batch_size)

            # --- 兼容两种返回类型：dict 或 tuple ---
            if isinstance(batch, dict):
                obs_concat  = batch['obs']
                act_joint   = batch['act']
                rew_b       = batch['reward']
                next_concat = batch['next_obs']
                done_b      = batch['terminal']
            else:
                # parl.utils.ReplayMemory: sample_batch -> (obs, act, reward, next_obs, terminal)
                obs_concat, act_joint, rew_b, next_concat, done_b = batch

            # --- 转成 numpy，保证连续内存与类型 ---
            import numpy as _np
            obs_concat  = _np.asarray(obs_concat,  dtype=_np.float32)
            act_joint   = _np.asarray(act_joint,   dtype=_np.int64)    # [B, n_agents]
            rew_b       = _np.asarray(rew_b,       dtype=_np.float32)  # [B] 或 [B,1]
            next_concat = _np.asarray(next_concat, dtype=_np.float32)
            done_b      = _np.asarray(done_b,      dtype=_np.float32)

            # —— 限次诊断打印（最多 3 次） —— #
            if not hasattr(self, "_dbg_qmix"):
                self._dbg_qmix = 0
            if self._dbg_qmix < 3:
                print("[MAAgent.learn/QMIX dbg] obs", getattr(obs_concat, "shape", type(obs_concat)),
                    "act", getattr(act_joint, "shape", type(act_joint)),
                    "rew", getattr(rew_b, "shape", type(rew_b)),
                    "next", getattr(next_concat, "shape", type(next_concat)),
                    "done", getattr(done_b, "shape", type(done_b)))
                try:
                    print("[MAAgent.learn/QMIX dbg] act head:", act_joint[:3])
                except Exception:
                    pass
                self._dbg_qmix += 1

            # 直接把 concat 后的 obs 与联合动作交给 QMIX（qmix.py 内部会切分 obs）
            loss = self.alg.learn(
                obs_n=obs_concat,
                act_n=act_joint,
                rewards=rew_b,
                next_obs_n=next_concat,
                dones=done_b,
            )
            # —— 统一安全转 float：优先 float(tensor)，退化到 numpy.item()
            try:
                return float(loss)
            except Exception:
                import numpy as _np
                return float(_np.asarray(loss).reshape(1)[0])


        # === END QMIX PATCH ===

        # ===== 其余算法保持原逻辑 =====
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
            algo_name = self.alg.__class__.__name__.lower()
            if algo_name == 'mappo':
                # —— MAPPO：直接按新签名传入 (obs_n, act_n, rewards, next_obs_n, dones)
                loss = self.alg.learn(
                    batch_obs_n,          # list[paddle.Tensor]
                    batch_act_n,          # list[paddle.Tensor]
                    rew_b,                # Tensor [B] or [B,1]
                    batch_obs_next_n,     # list[paddle.Tensor]
                    done_b,               # Tensor [B] or [B,1]
                )
            else:
                # —— MADDPG（保持原逻辑）：算 target_q 再 learn(obs_n, act_n, target_q)
                target_act_next_n = [
                    agents[i].alg.sample(batch_obs_next_n[i], use_target_model=True).detach()
                    for i in range(self.n)
                ]
                q_next    = self.alg.Q(batch_obs_next_n, target_act_next_n, use_target_model=True)
                target_q  = rew_b + self.alg.gamma * (1.0 - done_b) * q_next.detach()
                loss      = self.alg.learn(batch_obs_n, batch_act_n, target_q)
        else:
            # MADQN
            loss = 0.0
            for i in range(self.n):
                obs_i  = batch_obs_n[i]
                act_i  = batch_act_n[i].astype('int64')
                next_i = batch_obs_next_n[i]
                loss   = self.alg.learn(obs_i, act_i, rew_b, next_i, done_b)


        try:
            return float(loss)
        except Exception:
            import numpy as _np
            return float(_np.asarray(loss).reshape(1)[0])

