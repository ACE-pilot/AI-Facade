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
    特征自动从 image_features.csv 中随机选取（WGR,BH_mean,BS_mean,BL_mean,WH_mean,WS_mean,WL_mean）。
    奖励：中间步骤压力惩罚乘0.1，回合末压力惩罚不缩放并加二次函数额外奖励。
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
                 device="cpu"):
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
        self.df_feats = df[self.feature_cols]

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
            self.n_discrete = 3  # {0,1,2} 分别对应 -0.05, 0, +0.05
            self.single_action_space = spaces.Discrete(self.n_discrete)
        else:
            self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_spaces = {agent: self.single_action_space for agent in self.agents}

        # 观测空间：原始7维+当前7维+差值7维+自身动作1维+其他动作(num_agents-1)维
        obs_dim = self.num_agents*3 + 1 + (self.num_agents-1)
        self.single_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.observation_spaces = {agent: self.single_observation_space for agent in self.agents}

    def reset(self):
        """
        重置环境：随机采样，重置状态步数。
        返回每个智能体的观测：
          [orig, current, diff(0), self_act(0), other_act(0)]
        """
        # 采样新特征
        feat = self.df_feats.sample(n=1).values.flatten().astype(np.float32)
        self.original_features = feat
        self.current_features = np.copy(feat)
        self.current_step = 0

        # 初始观测
        obs = {}
        zero_diff = np.zeros(self.num_agents, dtype=np.float32)
        zero_self = np.zeros(1, dtype=np.float32)
        zero_other = np.zeros(self.num_agents-1, dtype=np.float32)
        for agent in self.agents:
            obs[agent] = np.concatenate([self.original_features,
                                         self.current_features,
                                         zero_diff,
                                         zero_self,
                                         zero_other])
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
                # ★ NEW: 索引三档 + 边界夹紧
                idx = int(raw_act)
                if idx < 0: idx = 0
                if idx > 2: idx = 2
                # ★ NEW: 三档直接映射：0→-step, 1→0, 2→+step
                if idx == 0:
                    adj = -self.discrete_step
                elif idx == 1:
                    adj = 0.0
                else:  # idx == 2
                    adj = +self.discrete_step

                if DEBUG_DISCRETE and i == 0:
                    print(f"[3-ACTION] step={self.current_step} idx={idx} adj={adj:+.5f}")
            else:
                contin = float(raw_act)
                adj = contin * self.continuous_step

            adjustments.append(adj)
            self.current_features[i] += adj


        # 压力预测
        scaled = self.scaler.transform(self.current_features.reshape(1,-1)).astype(np.float32)
        inp = torch.from_numpy(scaled).to(self.device)
        with torch.no_grad():
            pressure = float(self.pressure_model(inp).item())

        # 差值惩罚
        diffs = self.current_features - self.original_features
        excess = np.maximum(np.abs(diffs)-self.diff_threshold, 0.0)
        diff_penalty = -np.sum(excess)*0.1

        # 计算奖励
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
            obs_vec = np.concatenate([self.original_features,
                                      self.current_features,
                                      self.current_features-self.original_features,
                                      np.array([self_act],dtype=np.float32),
                                      np.array(other_acts,dtype=np.float32)])
            obs[agent]= obs_vec
            rewards[agent]= reward
            dones[agent]= done
            infos[agent]= {}
        return obs, rewards, dones, infos
