import gym
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
from joblib import load
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

class MLPRegressor(nn.Module):
    """
    与训练时一致的MLP回归模型结构，用于预测压力值
    """
    def __init__(self, in_dim=7, hidden_dims=[128, 64]):
        super(MLPRegressor, self).__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class FacadeEnv(gym.Env):
    """
    多智能体环境，每个智能体调整建筑立面特征。
    支持连续与离散动作模式：
      - 连续模式：动作范围[-1,1]，步长由 continuous_step 控制。
      - 离散模式：动作空间 Discrete，步长由 discrete_step 控制。
    特征由 CSV 指定列读取；当设置 scene_index 时，不再随机抽样，而是固定使用该场景行。
    奖励：中间步骤压力惩罚乘0.1，回合末压力惩罚不缩放并加二次函数额外奖励。
    统一硬边界：[0, 1]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 num_agents=7,
                 render_mode=False,
                 csv_path="image_features.csv",
                 model_path="facade_training_results/mlp_facade_model_reduced.pth",
                 scaler_path="facade_training_results/scaler.joblib",
                 diff_threshold=0.3,
                 max_steps=20,
                 continuous_step=0.1,
                 discrete_actions=False,
                 discrete_step=0.05,
                 device="cpu",
                 scene_index=27  # ★ 固定场景号（1-based传参，这里内部减1成0-based）；为 None 时回退为随机。
                 ):
        super(FacadeEnv, self).__init__()
        # 参数
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.diff_threshold = diff_threshold
        self.max_steps = max_steps
        self.current_step = 0
        self.continuous_step = continuous_step
        self.discrete_actions = discrete_actions
        self.discrete_step = discrete_step

        # 加载CSV特征表，仅保留7列指标
        df = pd.read_csv(csv_path, header=1)
        self.feature_cols = ["WGR","BH_mean","BS_mean","BL_mean","WH_mean","WS_mean","WL_mean"]
        self.df_feats = df[self.feature_cols].reset_index(drop=True).astype(np.float32)

        # 场景索引（固定场景号）；None 时表示随机
        self.scene_index = None if scene_index is None else int(scene_index) - 1
        self._check_scene_index()

        # 初始化特征
        self.original_features = np.zeros(self.num_agents, dtype=np.float32)
        self.current_features = np.copy(self.original_features)

        # 智能体列表
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # 加载Scaler和压力模型
        self.device = torch.device(device)
        self.scaler = load(scaler_path)
        self.pressure_model = MLPRegressor(in_dim=self.num_agents)
        self.pressure_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.pressure_model.to(self.device)
        self.pressure_model.eval()

        # 动作空间
        if self.discrete_actions:
            self.n_discrete = 3  # {0,1,2} 分别对应 -step, 0, +step
            self.single_action_space = spaces.Discrete(self.n_discrete)
        else:
            self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_spaces = {agent: self.single_action_space for agent in self.agents}

        # 观测空间：原始7维+当前7维+差值7维+自身动作1维+其他动作(num_agents-1)维
        obs_dim = self.num_agents*3 + 1 + (self.num_agents-1)
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_spaces = {agent: self.single_observation_space for agent in self.agents}

    # ===== 工具：校验/设置场景号 =====
    def _check_scene_index(self):
        if self.scene_index is None:
            return
        n = len(self.df_feats)
        if n == 0:
            raise ValueError("CSV 中没有可用样本行。")
        if not isinstance(self.scene_index, int):
            raise TypeError(f"scene_index 必须是 int 或 None，当前为 {type(self.scene_index)}")
        if not (0 <= self.scene_index < n):
            raise IndexError(f"scene_index 超界：应在 [0, {n-1}]，当前为 {self.scene_index}")

    def set_scene(self, idx: int):
        """在运行中切换固定场景号（0-based）；下次 reset 生效。"""
        self.scene_index = int(idx)
        self._check_scene_index()

    def _get_initial_features(self):
        """按固定场景或随机，返回 7 维特征 numpy.float32 数组。"""
        if self.scene_index is None:
            feat = self.df_feats.sample(n=1).values.flatten().astype(np.float32)
        else:
            feat = self.df_feats.iloc[self.scene_index].values.astype(np.float32)
        # 统一硬边界到 [0,1]
        return np.clip(feat, 0.0, 1.0)

    @staticmethod
    def _clip01(x: np.ndarray):
        return np.clip(x, 0.0, 1.0, out=x)

    def reset(self):
        """
        重置环境：若设置 scene_index 则固定场景；否则随机采样。
        返回每个智能体的观测：
          [orig, current, diff(0), self_act(0), other_act(0)]
        """
        feat = self._get_initial_features()
        self.original_features = feat.copy()
        self.current_features = feat.copy()
        # 再保险：裁一次（避免外部传入异常）
        self._clip01(self.original_features)
        self._clip01(self.current_features)

        self.current_step = 0

        # 初始观测
        obs = {}
        zero_diff = np.zeros(self.num_agents, dtype=np.float32)
        zero_self = np.zeros(1, dtype=np.float32)
        zero_other = np.zeros(self.num_agents-1, dtype=np.float32)
        concat = [self.original_features, self.current_features, zero_diff, zero_self, zero_other]
        obs_vec = np.concatenate(concat).astype(np.float32)
        for agent in self.agents:
            obs[agent] = obs_vec.copy()
        return obs

    def step(self, actions):
        self.current_step += 1
        adjustments = []
        DEBUG_DISCRETE = False  # 需要时置 True 看映射打印

        for i, agent in enumerate(self.agents):
            raw_act = actions[agent]
            if isinstance(raw_act, np.ndarray):
                raw_act = raw_act.item()

            if self.discrete_actions:
                # 三档 + 边界夹紧
                idx = int(raw_act)
                if idx < 0: idx = 0
                if idx > 2: idx = 2
                # 0→-step, 1→0, 2→+step
                if idx == 0:
                    adj = -self.discrete_step
                elif idx == 1:
                    adj = 0.0
                else:
                    adj = +self.discrete_step

                if DEBUG_DISCRETE and i == 0:
                    print(f"[3-ACTION] step={self.current_step} idx={idx} adj={adj:+.5f}")
            else:
                contin = float(raw_act)
                adj = contin * self.continuous_step

            adjustments.append(adj)
            self.current_features[i] += adj

        # ★★★ 统一硬边界裁剪到 [0,1]（在计算压力/奖励之前）
        self._clip01(self.current_features)

        # 压力预测
        scaled = self.scaler.transform(self.current_features.reshape(1,-1)).astype(np.float32)
        inp = torch.from_numpy(scaled).to(self.device)
        with torch.no_grad():
            pressure = float(self.pressure_model(inp).item())

        # 差值惩罚（越离初值超阈值，越惩罚）
        diffs = self.current_features - self.original_features
        excess = np.maximum(np.abs(diffs)-self.diff_threshold, 0.0)
        diff_penalty = -np.sum(excess)*0.1

        # 计算奖励（保留你的 extra 逻辑）
        if self.current_step < self.max_steps:
            reward = -pressure*0.1 + diff_penalty
            done = False
        else:
            extra = 10*(1-(pressure/0.5)**2) if pressure<=0.5 else 0.0
            reward = -pressure + diff_penalty + extra
            done = True

        # 构造输出
        obs, rewards, dones, infos = {}, {}, {}, {}
        for i, agent in enumerate(self.agents):
            self_act = adjustments[i]
            other_acts = [adjustments[j] for j in range(self.num_agents) if j!=i]
            obs_vec = np.concatenate([
                self.original_features,
                self.current_features,
                self.current_features - self.original_features,
                np.array([self_act], dtype=np.float32),
                np.array(other_acts, dtype=np.float32)
            ]).astype(np.float32)
            obs[agent]  = obs_vec
            rewards[agent] = reward
            dones[agent]   = done
            infos[agent]   = {}
        return obs, rewards, dones, infos
