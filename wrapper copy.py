import gym
from gym import spaces
import numpy as np

class MultiAgentWrapper(gym.Wrapper):
    """
    通用多智能体环境包装器，支持连续与离散动作模式。
    如果 discrete_actions=True，则动作空间为 Discrete(n_discrete)，
    映射到原始连续动作 [-1,1] 后再传给环境；否则保持 Box 连续空间。
    """
    def __init__(self, env, discrete_actions=False, step_size=0.05):
        super(MultiAgentWrapper, self).__init__(env)
        self.discrete_actions = discrete_actions
        self.step_size = step_size
        self.env = env

        # 智能体个数
        self.agents = list(self.env.action_spaces.keys())
        self.n = len(self.agents)

        # 设置动作空间
        if self.discrete_actions:
            # 允许动作范围 [-1,1]，步长 step_size
            self.n_discrete = int(2 / self.step_size) + 1
            self.action_space = {agent: spaces.Discrete(self.n_discrete) for agent in self.agents}
        else:
            # 连续模式下直接复用 env 的 action_spaces
            self.action_space = self.env.action_spaces

        # 设置观测空间
        self.observation_space = self.env.observation_spaces

        # 生成 agents_name 列表
        self.agent_names = self.agents.copy()

        # 计算 obs/act 维度列表
        self.obs_shape_n = [space.shape for space in self.observation_space.values()]
        self.act_shape_n = [self._get_act_dim(space) for space in self.action_space.values()]

    def _get_act_dim(self, space):
        if isinstance(space, spaces.Box):
            return space.shape[0]
        elif isinstance(space, spaces.Discrete):
            return space.n
        else:
            raise NotImplementedError(f"Unsupported action space: {space}")

    def reset(self):
        obs_dict = self.env.reset()
        return [obs_dict[name] for name in self.agent_names]

    def step(self, action_n):
        # 输入 action_n: list of actions for each agent
        # 构造原始动作字典
        action_dict = {}
        for i, raw_action in enumerate(action_n):
            agent = self.agent_names[i]
            if self.discrete_actions:
                idx = int(raw_action)
                # 映射到 [-1,1]
                contin_action = (idx / (self.n_discrete - 1)) * 2 - 1
            else:
                contin_action = np.clip(raw_action, 
                                        self.env.action_spaces[agent].low,
                                        self.env.action_spaces[agent].high)
            # 单维动作
            action_dict[agent] = np.array([contin_action], dtype=np.float32)

        # 与环境交互
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

        # 转换输出为列表
        obs_n = [obs_dict[name] for name in self.agent_names]
        reward_n = [reward_dict[name] for name in self.agent_names]
        done_n = [done_dict[name] for name in self.agent_names]
        info_n = [info_dict.get(name, {}) for name in self.agent_names]
        return obs_n, reward_n, done_n, info_n
