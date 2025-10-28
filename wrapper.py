# wrapper.py

import gym
from gym import spaces
import numpy as np

class MultiAgentWrapper(gym.Wrapper):
    """
    通用多智能体 Wrapper，支持连续与离散动作。
    - 连续模式：action_space 为 Box(-1,1)，值直接传给底层 env（env 内再乘 continuous_step）。
    - 离散模式（本项目三档）：action_space 为 Discrete(3)，
        索引 0/1/2 分别表示 -discrete_step / 0 / +discrete_step，
        索引原样传给底层 env，由 env 完成映射。
    """

    def __init__(self, env, discrete_actions=False, step_size=0.05):
        super(MultiAgentWrapper, self).__init__(env)
        self.env = env
        self.discrete_actions = discrete_actions
        self.step_size = step_size  # 连续分支用；离散分支由 env 的 discrete_step 决定

        # agent 名单
        if hasattr(env, 'agents'):
            self.agent_names = env.agents
        elif hasattr(env, 'agents_name'):
            self.agent_names = env.agents_name
        else:
            raise AttributeError("底层 env 必须定义 .agents 或 .agent_names")

        self.n = len(self.agent_names)

        # 观测空间
        self.observation_spaces = env.observation_spaces
        self.obs_shape_n = [space.shape for space in self.observation_spaces.values()]

        # 动作空间
        if self.discrete_actions:
            # ★ 离散三档：需要 env 暴露 n_discrete=3
            assert hasattr(env, 'n_discrete'), "FacadeEnv 在离散模式下需要暴露 n_discrete"
            assert int(env.n_discrete) >=2 , f"期望 n_discrete=3，实际为 {env.n_discrete}"
            self.single_action_space = spaces.Discrete(env.n_discrete)  # Discrete(3)
            self.act_shape_n = [env.n_discrete] * self.n
        else:
            self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.act_shape_n = [1] * self.n

        self.action_space = {name: self.single_action_space for name in self.agent_names}

    def reset(self):
        """返回：obs_n 列表，对应每个 agent 的观测向量"""
        obs_dict = self.env.reset()
        return [obs_dict[name] for name in self.agent_names]

    def step(self, action_n):
        """
        将上层给的动作列表转为底层 env 需要的格式后调用 env.step：
        - 离散：action_n[i] 是 0/1/2（或 [0]/[1]/[2]），原样传给 env（整数索引）。
        - 连续：裁剪到 [-1,1]，以 np.array([value]) 形式传给 env。
        """
        action_dict = {}
        for i, raw in enumerate(action_n):
            name = self.agent_names[i]
            # 解包列表/ndarray
            if isinstance(raw, (list, tuple, np.ndarray)):
                raw = raw[0]

            if self.discrete_actions:
                # ★ 关键：直接传“索引”给 env
                idx = int(raw)
                # 夹紧范围（避免上层越界）
                if idx < 0: idx = 0
                if idx > self.env.n_discrete - 1: idx = self.env.n_discrete - 1
                action_dict[name] = idx                    # 注意：传整数，而不是数组/浮点
            else:
                # 连续动作：裁剪到 [-1,1]
                cont = float(raw)
                cont = np.clip(cont, -1.0, 1.0)
                action_dict[name] = np.array([cont], dtype=np.float32)

        # 与底层交互
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

        # 转回列表格式
        obs_n  = [obs_dict[name]           for name in self.agent_names]
        rew_n  = [reward_dict[name]        for name in self.agent_names]
        done_n = [done_dict[name]          for name in self.agent_names]
        info_n = [info_dict.get(name, {})  for name in self.agent_names]

        return obs_n, rew_n, done_n, info_n
