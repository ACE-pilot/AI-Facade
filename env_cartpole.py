# env_cartpole.py
# Classic-reward CartPole (每步+1)，与用户自定义 MultiAgentWrapper (dict I/O) 完全兼容
from typing import Dict, Tuple, Optional, Any
import numpy as np
import gym

class CartPoleMultiEnv:
    """
    单/多智能体 CartPole 封装（默认单智能体，仅为快速检查训练效果）。
    与 MultiAgentWrapper 的契约：
      - .agents / .observation_spaces  (dict[name] = gym.spaces.Box(shape=(4,)))
      - 若离散：.n_discrete = 3 且 step(action_dict) 中 action ∈ {0,1,2}
      - reset() -> obs_dict
      - step(action_dict) -> (obs_dict, reward_dict, done_dict, info_dict)

    经典奖励：直接透传 Gym 的 reward（每存活一步 +1；倒杆/截断即终止）。
    不做任何团队均值/缩放/额外 shaping。
    """
    def __init__(
        self,
        n_agents: int = 1,                 # ← 单智能体
        episode_limit: int = 500,          # CartPole-v1 满回合
        team_reward: bool = False,         # ← 关闭团队均分，使用原始奖励
        team_done_when_any: bool = True,   # 多体时任一失败即终止（单体无差别）
        seed: Optional[int] = None,
        render_mode: Optional[str] = None, # 兼容老 gym：若不支持会自动退化
        discrete_actions: bool = True      # 与上层 wrapper 的 discrete_actions 对齐
    ):
        self.n_agents = int(n_agents)
        self.episode_limit = int(episode_limit)
        self.team_reward = bool(team_reward)
        self.team_done_when_any = bool(team_done_when_any)
        self.render_mode = render_mode
        self.discrete_actions = bool(discrete_actions)

        # agent 名称
        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        # 兼容老 gym 的构造（无 render_mode）
        def _make_cartpole(render_mode):
            try:
                return gym.make("CartPole-v1", render_mode=render_mode)
            except TypeError:
                return gym.make("CartPole-v1")

        # 创建子环境
        self.envs = []
        for i in range(self.n_agents):
            env = _make_cartpole(render_mode=self.render_mode)
            if seed is not None:
                try:
                    env.action_space.seed(seed + i)
                    env.observation_space.seed(seed + i)
                except Exception:
                    pass
            self.envs.append(env)

        # 观测空间（dict 形式）
        obs_space = self.envs[0].observation_space  # shape=(4,)
        self.observation_spaces: Dict[str, Any] = {name: obs_space for name in self.agents}

        # 离散三档（供 wrapper 断言/转发）；CartPole 原生动作数为 2（左/右）
        if self.discrete_actions:
            self.n_discrete: int = 2
        self.n_actions = 2
        self.obs_shape = int(obs_space.shape[0])    # 4
        self.state_shape = self.obs_shape * self.n_agents

        # 记录
        self._t = 0
        self._seed_base = seed
        self._last_obs_dict: Optional[Dict[str, np.ndarray]] = None
        self._last_discrete: Dict[str, int] = {name: 1 for name in self.agents}  # “中档复用”的上一次档位

    # ===== 常见 MARL 接口补充 =====
    def get_state(self, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        if obs_dict is None:
            obs_dict = self._last_obs_dict
        obs_list = [obs_dict[name] for name in self.agents]
        return np.concatenate(obs_list, axis=0).astype(np.float32)

    def get_env_info(self) -> Dict[str, Any]:
        return dict(
            n_agents=self.n_agents,
            n_actions=(self.n_discrete if self.discrete_actions else self.n_actions),
            obs_shape=self.obs_shape,
            state_shape=self.state_shape,
            episode_limit=self.episode_limit,
            is_discrete_action=self.discrete_actions
        )

    # ===== 与 wrapper 的契约：reset/step 返回字典 =====
    def reset(self) -> Dict[str, np.ndarray]:
        self._t = 0
        obs_dict: Dict[str, np.ndarray] = {}
        for i, (name, env) in enumerate(zip(self.agents, self.envs)):
            out = env.reset(seed=(self._seed_base + i) if self._seed_base is not None else None)
            # gymnasium: (obs, info) ; 老 gym: obs
            obs = out[0] if (isinstance(out, tuple) and len(out) == 2) else out
            obs_dict[name] = np.asarray(obs, dtype=np.float32)
            self._last_discrete[name] = 1  # 中档
        self._last_obs_dict = obs_dict
        return obs_dict

    def step(self, action_dict: Dict[str, Any]) -> Tuple[
        Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]
    ]:
        self._t += 1
        obs_dict: Dict[str, np.ndarray] = {}
        reward_dict: Dict[str, float] = {}
        done_flags: Dict[str, bool] = {}
        info_dict: Dict[str, Any] = {}

        for name, env in zip(self.agents, self.envs):
            cp_action = self._map_action(name, action_dict[name])  # → {0,1}
            out = env.step(int(cp_action))
            # gymnasium: (obs, reward, terminated, truncated, info) ; 老 gym: (obs, reward, done, info)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = out

            obs_dict[name] = np.asarray(obs, dtype=np.float32)
            reward_dict[name] = float(reward)  # 经典奖励：直接透传
            done_flags[name] = bool(done)
            info_dict[name] = {"discrete_idx": self._last_discrete.get(name, 1)}

        # 不做团队奖励加工（team_reward=False 时不会进入）
        if self.team_reward:
            mean_r = float(np.mean([reward_dict[n] for n in self.agents]))
            for n in self.agents:
                reward_dict[n] = mean_r

        # 终止条件：团队 or 个体 + 回合步数上限
        if self.team_done_when_any:
            any_done = any(done_flags.values()) or (self._t >= self.episode_limit)
            done_dict = {n: any_done for n in self.agents}
        else:
            done_dict = {n: (done_flags[n] or (self._t >= self.episode_limit)) for n in self.agents}

        self._last_obs_dict = obs_dict
        return obs_dict, reward_dict, done_dict, info_dict

    def _map_action(self, name: str, raw: Any) -> int:
        """离散：0->左, 1->右；连续：按符号阈值→{0,1}"""
        # 连续输入
        if isinstance(raw, (list, tuple, np.ndarray)):
            val = float(np.asarray(raw).reshape(-1)[0])
            return 1 if val > 0.0 else 0
        # 离散输入
        idx = int(raw)
        if idx < 0: idx = 0
        if idx > 1: idx = 1
        return idx


    # ===== 渲染/关闭 =====
    def render(self):
        try:
            self.envs[0].render()
        except Exception:
            pass

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
