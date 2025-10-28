# simple_agent.py  —— 以旧版为模板，MAPPO 定向改动 + MADQN 最小增补
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
          discrete_actions (bool): 是否为离散动作算法（如 MADQN/QMIX/MAPPO-离散）
          step_size        (float): 离散动作步长（仅在 wrapper 中使用）
        """
        # 保存参数
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

        # === 与旧版一致：用是否存在 Q(...) 区分值函数（MADQN/QMIX）与其他算法 ===
        self.is_value_based = (not hasattr(algorithm, 'Q'))

        # === MAPPO 判定（仅用于“定向改动”） ===
        algo_cls_name = type(algorithm).__name__.lower()
        is_mappo_algo = (algo_cls_name == 'mappo')

        # === 初始化 replay memory ===
        obs_shape = obs_dim_n[self.agent_index]
        obs_dim   = int(np.prod(obs_shape))
        raw_act_dim = int(act_dim_n[self.agent_index])

        # 【MADQN 最小增补】值函数按“索引一列”存；
        # 【MAPPO 定向】若 MAPPO 且为离散动作，同样按“索引一列”存，避免 dtype/形状错配。
        if self.is_value_based or (is_mappo_algo and self.discrete_actions):
            store_act_dim = 1
        else:
            store_act_dim = raw_act_dim

        self.rpm = ReplayMemory(
            max_size=int(1e6),   # 旧版就是 1e6
            obs_dim=obs_dim,
            act_dim=store_act_dim
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

        # 【MADQN 最小增补】离散 ε-greedy 超参数 + 学习节奏（仅值函数用）
        self.eps_start = 1.0
        self.eps_end   = 0.05
        self.eps_decay_steps = 20000
        self.steps_done = 0

        self.learn_every = 20      # MADQN: 每 20 步学一次
        self.warmup_mult = 5       # MADQN: warmup = batch_size * 5

    # ---- MAPPO 判定（基于已绑定的 self.alg）----
    def _is_mappo(self):
        try:
            return self.alg.__class__.__name__.lower() == 'mappo'
        except Exception:
            return False

    def predict(self, obs):
        # ====== MAPPO 专用：策略预测（不受 MADQN 的 argmax 影响） ======
        if self._is_mappo():
            obs_shape = self.obs_dim_n[self.agent_index]
            x = paddle.to_tensor(np.asarray(obs, dtype='float32').reshape(1, *obs_shape), dtype='float32')
            act = self.alg.predict(x)
            if isinstance(act, paddle.Tensor):
                act = act.detach().cpu().numpy()
            if self.discrete_actions:
                return [int(np.asarray(act).reshape(-1)[0])]
            else:
                return np.asarray(act).flatten()

        # ====== 值函数：返回 argmax 索引 list（MADQN/QMIX 用） ======
        if self.is_value_based:
            arr    = np.array(obs, dtype='float32').reshape(1, -1)
            tensor = paddle.to_tensor(arr, dtype='float32')
            out    = self.alg.predict(tensor)
            q      = out.detach().cpu().numpy().reshape(-1)
            return [int(np.argmax(q))]

        # ====== 连续（旧版路径） ======
        arr    = np.array(obs, dtype='float32').reshape(1, -1)
        tensor = paddle.to_tensor(arr, dtype='float32')
        out    = self.alg.predict(tensor)
        return out.detach().cpu().numpy().flatten()

    def sample(self, obs, use_target_model=False):
        """
        老板旧版行为对齐：
        - 对于连续动作算法（MADDPG / 其它非值函数连续）：按每个体原始观测形状 reshape -> (1, *obs_shape)
          再送入 alg.sample；直接返回连续动作（不做 argmax）。
        - 【MAPPO 定向】无论离散/连续，都走策略采样（不受 MADQN 的 ε-greedy 影响）。
        - 【MADQN 最小增补】对于值函数算法：flatten -> (1, -1) 做 ε-greedy 选动作索引。
        """
        # ====== MAPPO 专用：策略采样 ======
        if self._is_mappo():
            obs_shape = self.obs_dim_n[self.agent_index]
            x = paddle.to_tensor(np.asarray(obs, dtype='float32').reshape(1, *obs_shape), dtype='float32')
            act = self.alg.sample(x)  # MAPPO.sample 忽略 use_target_model
            if isinstance(act, paddle.Tensor):
                act = act.detach().cpu().numpy()
            if self.discrete_actions:
                return [int(np.asarray(act).reshape(-1)[0])]
            else:
                return np.asarray(act).flatten()

        # ====== 原有连续路径（MADDPG 等） ======
        is_continuous_algo = (not self.discrete_actions) and (not self.is_value_based)
        if is_continuous_algo:
            obs_arr   = np.array(obs, dtype='float32')
            obs_shape = self.obs_dim_n[self.agent_index]
            obs_tensor = paddle.to_tensor(obs_arr.reshape(1, *obs_shape), dtype='float32')
            try:
                act_out = self.alg.sample(obs_tensor, use_target_model=use_target_model)
            except TypeError:
                act_out = self.alg.sample(obs_tensor)
            if isinstance(act_out, paddle.Tensor):
                act_out = act_out.detach().cpu().numpy()
            return act_out.detach().cpu().numpy().flatten() if isinstance(act_out, paddle.Tensor) else np.asarray(act_out).flatten()

        # ====== 值函数（MADQN/QMIX）：ε-greedy ======
        arr    = np.array(obs, dtype='float32').reshape(1, -1)
        tensor = paddle.to_tensor(arr, dtype='float32')
        try:
            qvals = self.alg.sample(tensor, use_target_model=use_target_model)
        except TypeError:
            qvals = self.alg.sample(tensor)

        q = qvals.detach().cpu().numpy().reshape(-1) if isinstance(qvals, paddle.Tensor) \
            else np.asarray(qvals).reshape(-1)

        act_dim = int(self.act_dim_n[self.agent_index])
        eps = max(self.eps_end,
                  self.eps_start - (self.eps_start - self.eps_end) *
                  min(1.0, self.steps_done / float(self.eps_decay_steps)))
        self.steps_done += 1

        a = np.random.randint(act_dim) if np.random.rand() < eps else int(np.argmax(q))
        return [a]

    def add_experience(self, obs, act, reward, next_obs, terminal):
        # —— 旧版：直接 flatten obs/next_obs 存入回放
        obs_flat  = np.asarray(obs, dtype='float32').flatten()
        next_flat = np.asarray(next_obs, dtype='float32').flatten()

        # 【MADQN 最小增补】：值函数按“动作索引一列”存
        if self.is_value_based:
            idx = int(np.asarray(act).reshape(-1)[0]) if isinstance(act, (list, tuple, np.ndarray)) else int(act)
            act_stored = np.array([idx], dtype=np.int64)
        # 【MAPPO 定向】：MAPPO-离散同样按索引一列存，MAPPO-连续按旧版原样存
        elif self._is_mappo() and self.discrete_actions:
            idx = int(np.asarray(act).reshape(-1)[0]) if isinstance(act, (list, tuple, np.ndarray)) else int(act)
            act_stored = np.array([idx], dtype=np.int64)
        else:
            act_stored = np.asarray(act, dtype='float32')

        self.rpm.append(obs_flat, act_stored, reward, next_flat, terminal)

    def learn(self, agents):
        # === QMIX PATCH: 若注入了集中式回放（shared_rpm），保持原有路径 ===
        if self.algo_name == 'qmix' and self.shared_rpm is not None:
            self.global_train_step += 1
            # 更高频的更新 + 更短的 warmup，单体/小环境更容易起势
            if self.global_train_step % 20 != 0:
                return 0.0
            if self.shared_rpm.size() <= self.batch_size * 5:
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
                obs_concat, act_joint, rew_b, next_concat, done_b = batch

            import numpy as _np
            obs_concat  = _np.asarray(obs_concat,  dtype=_np.float32)
            act_joint   = _np.asarray(act_joint,   dtype=_np.int64)
            rew_b       = _np.asarray(rew_b,       dtype=_np.float32)
            next_concat = _np.asarray(next_concat, dtype=_np.float32)
            done_b      = _np.asarray(done_b,      dtype=_np.float32)

            loss = self.alg.learn(
                obs_n=obs_concat,
                act_n=act_joint,
                rewards=rew_b,
                next_obs_n=next_concat,
                dones=done_b,
            )
            try:
                return float(loss)
            except Exception:
                import numpy as _np
                return float(_np.asarray(loss).reshape(1)[0])

        # === 【离散值函数】：MADQN / QMIX ===
        if self.is_value_based:
            # 判定是否 QMIX（有 mixer 的集中式值函数）
            algo_name = type(self.alg).__name__.upper()
            is_qmix = ('QMIX' in algo_name) or hasattr(self.alg, 'mix')

            if is_qmix:
                # —— QMIX：全体共享同一批 idx，组装联合 batch，一次性更新 —— #
                if self.rpm.size() <= self.batch_size * self.warmup_mult:
                    return 0.0

                idxes = self.rpm.make_index(self.batch_size)

                batch_obs_n, batch_act_n, batch_obs_next_n = [], [], []
                for i in range(self.n):
                    o, a, _, o2, _ = agents[i].rpm.sample_batch_by_index(idxes)
                    # 统一形状与 dtype
                    o  = o.reshape(-1, *self.obs_dim_n[i]).astype('float32')
                    o2 = o2.reshape(-1, *self.obs_dim_n[i]).astype('float32')
                    a  = np.asarray(a).reshape(-1).astype('int64')
                    batch_obs_n.append(o)
                    batch_act_n.append(a)
                    batch_obs_next_n.append(o2)

                # 团队回报与 done（从当前体回放取一份即可）
                _, _, rew_b, _, done_b = self.rpm.sample_batch_by_index(idxes)
                rew_b  = np.asarray(rew_b, dtype=np.float32).reshape(-1)
                done_b = np.asarray(done_b, dtype=np.float32).reshape(-1)

                # 仅让一个体触发优化，避免重复反向
                if self.agent_index != 0:
                    return 0.0

                joint_act = np.stack(batch_act_n, axis=1).astype('int64')  # [B, n_agents]

                loss = self.alg.learn(
                    obs_n=batch_obs_n,            # list of [B, obs_i]
                    act_n=joint_act,              # [B, n_agents] (int64)
                    rewards=rew_b,                # [B]
                    next_obs_n=batch_obs_next_n,  # list of [B, obs_i]
                    dones=done_b                  # [B]
                )
                try:
                    return float(loss)
                except Exception:
                    return float(np.asarray(loss).reshape(1)[0])

            else:
                # —— MADQN：沿用你原有的本地回放 + 学习节奏 —— #
                self.global_train_step += 1
                if self.global_train_step % self.learn_every != 0:
                    return 0.0
                if self.rpm.size() <= self.batch_size * self.warmup_mult:
                    return 0.0

                batch = self.rpm.sample_batch(self.batch_size)
                if isinstance(batch, dict):
                    obs      = batch['obs']
                    act      = batch['act'][:, 0].astype('int64')  # 索引一列 -> 压成 (B,)
                    reward   = batch['reward'].astype('float32')
                    next_obs = batch['next_obs']
                    terminal = batch['terminal'].astype('float32')
                else:
                    obs, act, reward, next_obs, terminal = batch
                    act      = np.asarray(act)[:, 0].astype('int64')
                    reward   = np.asarray(reward, dtype=np.float32)
                    terminal = np.asarray(terminal, dtype=np.float32)

                loss = self.alg.learn(obs, act, reward, next_obs, terminal)
                return float(loss) if isinstance(loss, paddle.Tensor) else float(loss)

        # ===== 连续（MADDPG/MAPPO）保持旧版逻辑，MAPPO 分支仅“定向处理 dtype/形状” =====
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

        # ===== MADDPG/MAPPO 统一入口（与旧版一致），仅在 MAPPO 分支规范 act 的 dtype/形状 =====
        if hasattr(self.alg, 'Q'):
            algo_name = self.alg.__class__.__name__.lower()

            if algo_name == 'mappo':
                # obs/next_obs: float32
                batch_obs_n_t      = [paddle.to_tensor(x, dtype='float32') for x in batch_obs_n]
                batch_obs_next_n_t = [paddle.to_tensor(x, dtype='float32') for x in batch_obs_next_n]

                # act_n：
                # - MAPPO-离散：需要 int64，形状 [B]
                # - MAPPO-连续：float32
                if self.discrete_actions:
                    norm_act_n = []
                    for a in batch_act_n:
                        a_np = np.asarray(a)
                        if a_np.ndim > 1 and a_np.shape[-1] == 1:
                            a_np = a_np.reshape(-1)
                        norm_act_n.append(paddle.to_tensor(a_np.astype('int64')))
                    batch_act_n_t = norm_act_n
                else:
                    batch_act_n_t = [paddle.to_tensor(x, dtype='float32') for x in batch_act_n]

                loss = self.alg.learn(
                    batch_obs_n_t,          # list[paddle.Tensor] float32
                    batch_act_n_t,          # list[paddle.Tensor] int64(离散)/float32(连续)
                    rew_b,                  # Tensor [B] or [B,1]
                    batch_obs_next_n_t,     # list[paddle.Tensor]
                    done_b,                 # Tensor [B] or [B,1]
                )

            else:
                # —— MADDPG：算 target_q 再 learn(obs_n, act_n, target_q)（旧版）
                batch_obs_n_t      = [paddle.to_tensor(x, dtype='float32') for x in batch_obs_n]
                batch_act_n_t      = [paddle.to_tensor(x, dtype='float32') for x in batch_act_n]
                batch_obs_next_n_t = [paddle.to_tensor(x, dtype='float32') for x in batch_obs_next_n]

                target_act_next_n = [
                    agents[i].alg.sample(batch_obs_next_n_t[i], use_target_model=True).detach()
                    for i in range(self.n)
                ]
                q_next    = self.alg.Q(batch_obs_next_n_t, target_act_next_n, use_target_model=True)
                target_q  = rew_b + self.alg.gamma * (1.0 - done_b) * q_next.detach()
                loss      = self.alg.learn(batch_obs_n_t, batch_act_n_t, target_q)
        else:
            # （理论上不会走到；保底）
            return 0.0

        try:
            return float(loss)
        except Exception:
            return float(np.asarray(loss).reshape(1)[0])
